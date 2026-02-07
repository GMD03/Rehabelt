#include <WiFi.h>
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <Adafruit_BNO055.h>
#include <Adafruit_Sensor.h>
#include <utility/imumaths.h>
#include <Wire.h>
#include <EloquentTinyML.h>
#include "model_data.h"

// CONFIGURATIONS
#define CS_PIN 5              
#define WINDOW_SIZE 100       
#define NUMBER_OF_INPUTS 600  
#define NUMBER_OF_OUTPUTS 4   
#define TENSOR_ARENA_SIZE 32 * 1024 
#define WIFI_SWITCH_PIN 35    
#define CONSENSUS_WINDOWS 3
#define SLIDE_STEP 20         
#define CONF_THRESH 0.85

// LED & MOTOR PINS
#define STATUS_LED_PIN 14     
#define WIFI_LED_PIN 15       
#define VIBRATION_MOTOR_PIN 17 

// WIFI CREDENTIALS
const char* ap_ssid = "ESP32-Rehab-Device";
const char* ap_password = "rehab1234";

// UPDATED MODEL
float MODEL_MEAN[] = { -0.470197, -7.90275569, -0.60975662, 0.03853848, -0.05213514, 0.0221224 };
float MODEL_SCALE[] = { 2.20417416, 3.3544568, 4.28097414, 0.40223686, 0.67285342, 0.313677 };


// OBJECTS & GLOBALS
WiFiServer server(80);
bool wifiMode = false;

// SPI PINS FOR ESP32-S3
#define MOSI_PIN 11
#define MISO_PIN 13  
#define SCK_PIN  12
SPIClass sd_spi = SPIClass(HSPI);

Adafruit_BNO055 mpu = Adafruit_BNO055(55, 0x28);
Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> classifier;

// BUFFERS
float sensorBuffer[WINDOW_SIZE][6]; 
float rawBuffer[WINDOW_SIZE][6];    
float flatInput[600];
float prediction[4];

// STATE VARIABLES
int sampleIndex = 0;
unsigned long lastFullCycleTime = 0;
const unsigned long MIN_CYCLE_INTERVAL = 3000; 
unsigned long lastInferenceTime = 0;
const int inferenceInterval = 1000;
bool isCalibrating = true;
unsigned long lastBlinkTime = 0;
bool blinkState = false;

// VIBRATION CONTROL
bool vibrationActive = false;
unsigned long vibrationStartTime = 0;
const unsigned long VIBRATION_DURATION = 1000;

// HISTORY TRACKING
#define WALKING_HISTORY_SIZE 10
bool walkingHistory[WALKING_HISTORY_SIZE] = {false};
int walkingHistoryIndex = 0;

// TIME VARIABLES
time_t now;
struct tm timeinfo;


// FUNCTION DECLARATIONS
void runWifiMode();
void runNormalMode();
void checkSwitch();
void updateStatusLED();
void checkCalibration();
void getRealTimeSensorData(float* ax, float* ay, float* az, float* gx, float* gy, float* gz);
bool isMotionSignificant();
int runModel();
void logToSD(String activity);
void calculateGaitMetrics();
void logGaitToSD(String datetime, String metric, String value);
void initInternalRTC();
String getDateTimeString();
void startWiFiHotspot();
void stopWiFi();
bool deleteAllFiles();


// SETUP
void setup() {
    Serial.begin(115200);
    delay(3000);
    
    pinMode(WIFI_SWITCH_PIN, INPUT_PULLUP);
    pinMode(STATUS_LED_PIN, OUTPUT);
    pinMode(WIFI_LED_PIN, OUTPUT);
    pinMode(VIBRATION_MOTOR_PIN, OUTPUT);
    
    digitalWrite(STATUS_LED_PIN, LOW);
    digitalWrite(WIFI_LED_PIN, LOW);
    digitalWrite(VIBRATION_MOTOR_PIN, LOW);
    
    Serial.println("\n\n=== ESP32-S3 Rehab Device Initialization ===");

    // INITIALIZE SD CARD
    Serial.print("Initializing SD card...");
    sd_spi.begin(SCK_PIN, MISO_PIN, MOSI_PIN, CS_PIN);
    if (!SD.begin(CS_PIN, sd_spi, 4000000)) {
        Serial.println("FAILED!");
    } else {
        Serial.println("OK");
        if (!SD.exists("/exercise.csv")) {
            File f = SD.open("/exercise.csv", FILE_WRITE);
            if(f) { f.println("Date & Time,Exercise"); f.close(); }
        }
        if (!SD.exists("/gait.csv")) {
            File f = SD.open("/gait.csv", FILE_WRITE);
            if(f) { f.println("Date & Time,Metric,Value"); f.close(); }
        }
    }

    // INITIALIZE RTC
    initInternalRTC();

    // INITIALIZE BNO055 MODULE
    Serial.print("Initializing BNO055...");
    Wire.begin(8, 9);
    if (!mpu.begin()) {
        Serial.println("FAILED! Check wiring.");
    } else {
        Serial.println("OK");
        mpu.setExtCrystalUse(true);
    }

    // Init AI Model
    Serial.print("Loading ML model...");
    if (!classifier.begin(model_data)) {
        Serial.println("FAILED! Model Error.");
    } else {
        Serial.println("OK");
    }
}


// MAIN LOOP
void loop() {
    checkSwitch();
    
    // VIBRATION TIMER
    if (vibrationActive && (millis() - vibrationStartTime >= VIBRATION_DURATION)) {
        vibrationActive = false;
        digitalWrite(VIBRATION_MOTOR_PIN, LOW);
    }

    if (wifiMode) {
        runWifiMode();
    } else {
        runNormalMode();
    }
}

// MOTION GATE WITH MODEL
bool isMotionSignificant() {
    float totalMovement = 0;
    float maxGyro = 0;
    float maxAccChange = 0;
    
    for (int i = 1; i < WINDOW_SIZE; i++) {
        float accChange = abs(rawBuffer[i][0] - rawBuffer[i-1][0]) +
                          abs(rawBuffer[i][1] - rawBuffer[i-1][1]) +
                          abs(rawBuffer[i][2] - rawBuffer[i-1][2]);
        
        if (accChange > maxAccChange) maxAccChange = accChange;
        
        float gyroMag = sqrt(pow(rawBuffer[i][3], 2) + pow(rawBuffer[i][4], 2) + pow(rawBuffer[i][5], 2));
        if (gyroMag > maxGyro) maxGyro = gyroMag;
        
        totalMovement += accChange + gyroMag;
    }
    
    float avgMovement = totalMovement / (WINDOW_SIZE - 1);
    //--------!!!!---------
    bool isSignificant = (avgMovement > 0.1) && 
                         (maxAccChange > 0.2) && 
                         (maxGyro > 0.8); 

    if (isSignificant) Serial.println("Motion Gate: OPEN (Movement Detected)");
    
    return isSignificant;
}

void runNormalMode() {
    updateStatusLED();
    checkCalibration();
    
    if (isCalibrating) { delay(100); return; }
    
    float ax, ay, az, gx, gy, gz;
    getRealTimeSensorData(&ax, &ay, &az, &gx, &gy, &gz);

    rawBuffer[sampleIndex][0] = ax;
    rawBuffer[sampleIndex][1] = ay;
    rawBuffer[sampleIndex][2] = az;
    rawBuffer[sampleIndex][3] = gx;
    rawBuffer[sampleIndex][4] = gy;
    rawBuffer[sampleIndex][5] = gz;


    sensorBuffer[sampleIndex][0] = (ax - MODEL_MEAN[0]) / MODEL_SCALE[0];
    sensorBuffer[sampleIndex][1] = (ay - MODEL_MEAN[1]) / MODEL_SCALE[1];
    sensorBuffer[sampleIndex][2] = (az - MODEL_MEAN[2]) / MODEL_SCALE[2];
    sensorBuffer[sampleIndex][3] = (gx - MODEL_MEAN[3]) / MODEL_SCALE[3];
    sensorBuffer[sampleIndex][4] = (gy - MODEL_MEAN[4]) / MODEL_SCALE[4];
    sensorBuffer[sampleIndex][5] = (gz - MODEL_MEAN[5]) / MODEL_SCALE[5];

    sampleIndex++;

    if (sampleIndex >= WINDOW_SIZE) {
        unsigned long currentTime = millis();
        
        if (currentTime - lastFullCycleTime < MIN_CYCLE_INTERVAL) {

            memmove(sensorBuffer, sensorBuffer + SLIDE_STEP, sizeof(float) * (WINDOW_SIZE - SLIDE_STEP) * 6);
            memmove(rawBuffer, rawBuffer + SLIDE_STEP, sizeof(float) * (WINDOW_SIZE - SLIDE_STEP) * 6);
            sampleIndex = WINDOW_SIZE - SLIDE_STEP;
            delay(20);
            return;
        }

        if (!isMotionSignificant()) {
            Serial.println("Motion Gate: CLOSED (User is still)");
            memmove(sensorBuffer, sensorBuffer + SLIDE_STEP, sizeof(float) * (WINDOW_SIZE - SLIDE_STEP) * 6);
            memmove(rawBuffer, rawBuffer + SLIDE_STEP, sizeof(float) * (WINDOW_SIZE - SLIDE_STEP) * 6);
            sampleIndex = WINDOW_SIZE - SLIDE_STEP;
            lastFullCycleTime = currentTime; 
            delay(20);
            return;
        }


        lastFullCycleTime = currentTime;
        int finalClass = runModel();
        
        String status = "UNRECOGNIZED";
        if (finalClass == 0) status = "Walking";
        else if (finalClass == 1) status = "Sit-to-Stand";
        else if (finalClass == 2) status = "Supine-to-Sit";

        Serial.print(" RESULT: "); Serial.println(status);

        walkingHistory[walkingHistoryIndex] = (finalClass == 0);
        walkingHistoryIndex = (walkingHistoryIndex + 1) % WALKING_HISTORY_SIZE;

        int prevIdx = (walkingHistoryIndex - 1 + WALKING_HISTORY_SIZE) % WALKING_HISTORY_SIZE;
        int prevPrevIdx = (walkingHistoryIndex - 2 + WALKING_HISTORY_SIZE) % WALKING_HISTORY_SIZE;
        if (walkingHistory[prevPrevIdx] && !walkingHistory[prevIdx]) {
            Serial.println("🚶 Walking Ended -> Computing Metrics...");
            calculateGaitMetrics();
        }

        if (finalClass != 3) {
            logToSD(status);
        } else {
            vibrationActive = true;
            vibrationStartTime = millis();
            digitalWrite(VIBRATION_MOTOR_PIN, HIGH);
        }

        memmove(sensorBuffer, sensorBuffer + SLIDE_STEP, sizeof(float) * (WINDOW_SIZE - SLIDE_STEP) * 6);
        memmove(rawBuffer, rawBuffer + SLIDE_STEP, sizeof(float) * (WINDOW_SIZE - SLIDE_STEP) * 6);
        sampleIndex = WINDOW_SIZE - SLIDE_STEP;
    }
    
    delay(20); 
}

int runModel() {
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 6; j++) {
            flatInput[i * 6 + j] = sensorBuffer[i][j];
        }
    }

    if (!classifier.predict(flatInput, prediction)) {
        Serial.println(" Inference Failed");
        return 3;
    }

    int maxIndex = 0;
    float maxVal = prediction[0];
    for (int i = 1; i < 4; i++) {
        if (prediction[i] > maxVal) {
            maxVal = prediction[i];
            maxIndex = i;
        }
    }
    
    Serial.print("Conf: [");
    Serial.print(prediction[0]); Serial.print(", ");
    Serial.print(prediction[1]); Serial.print(", ");
    Serial.print(prediction[2]); Serial.print(", ");
    Serial.print(prediction[3]); Serial.println("]");
    
    return maxIndex;
}


// SENSORS + ACTUATORS
void getRealTimeSensorData(float* ax, float* ay, float* az, float* gx, float* gy, float* gz) {
    imu::Vector<3> accel = mpu.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
    imu::Vector<3> gyro = mpu.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
    *ax = accel.x(); *ay = accel.y(); *az = accel.z();
    *gx = gyro.x();  *gy = gyro.y();  *gz = gyro.z();
}

void updateStatusLED() {
    if (isCalibrating) {
        if (millis() - lastBlinkTime > 500) {
            lastBlinkTime = millis();
            blinkState = !blinkState;
            digitalWrite(STATUS_LED_PIN, blinkState);
        }
    } else {
        digitalWrite(STATUS_LED_PIN, HIGH);
    }
}

void checkCalibration() {
    if (!isCalibrating) return;
    static unsigned long lastCheck = 0;
    if (millis() - lastCheck > 1000) {
        lastCheck = millis();
        uint8_t sys, gyro, accel, mag;
        mpu.getCalibration(&sys, &gyro, &accel, &mag);
        Serial.printf("Calib: Sys=%d Gyro=%d Accel=%d Mag=%d\n", sys, gyro, accel, mag);
        if (mag >= 2 && sys >= 2) isCalibrating = false;
    }
}

void initInternalRTC() {
    configTime(0, 0, "pool.ntp.org");
    setenv("TZ", "PST8PDT,M3.2.0,M11.1.0", 1); //TIMEZONE
    tzset();
}

String getDateTimeString() {
    time(&now);
    localtime_r(&now, &timeinfo);
    char buffer[20];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M", &timeinfo);
    return String(buffer);
}

void logToSD(String activity) {
    File file = SD.open("/exercise.csv", FILE_APPEND);
    if(file){
        file.print(getDateTimeString());
        file.print(",");
        file.println(activity);
        file.close();
        Serial.println("Logged to SD: " + activity);
    }
}

void logGaitToSD(String datetime, String metric, String value) {
    File file = SD.open("/gait.csv", FILE_APPEND);
    if(file){
        file.print(datetime);
        file.print(",");
        file.print(metric);
        file.print(",");
        file.println(value);
        file.close();
    }
}

void calculateGaitMetrics() {
    float sumAccX = 0;
    float maxAccY = -999, minAccY = 999;
    
    for (int i = 0; i < WINDOW_SIZE; i++) {
        float vert = rawBuffer[i][1] - 9.8; 
        if (vert > maxAccY) maxAccY = vert;
        if (vert < minAccY) minAccY = vert;
        sumAccX += rawBuffer[i][0];
    }
    
    int steps = 0;
    float threshold = (maxAccY - minAccY) * 0.4;
    for (int i = 1; i < WINDOW_SIZE - 1; i++) {
        float v = rawBuffer[i][1] - 9.8;
        if (v > rawBuffer[i-1][1]-9.8 && v > rawBuffer[i+1][1]-9.8 && v > threshold) {
            steps++;
            i += 5; // Debounce
        }
    }
    
    float seconds = WINDOW_SIZE / 50.0;
    float cadence = (steps / seconds) * 60.0;
    float meanAccX = sumAccX / WINDOW_SIZE;
    float varianceX = 0;
    for (int i = 0; i < WINDOW_SIZE; i++) varianceX += pow(rawBuffer[i][0] - meanAccX, 2);
    float lateralSway = sqrt(varianceX / WINDOW_SIZE) * 2.0 * 100; // cm estimate
    float stepLength = 0.4 + (0.15 * (cadence / 100.0)); // meters estimate

    String dt = getDateTimeString();
    logGaitToSD(dt, "Cadence", String(cadence, 1));
    logGaitToSD(dt, "Lateral Sway", String(lateralSway, 1));
    logGaitToSD(dt, "Step Length", String(stepLength * 100, 1));
}


// DASHBOARD
void runWifiMode() {
    digitalWrite(STATUS_LED_PIN, LOW);
    digitalWrite(WIFI_LED_PIN, HIGH);
    
    WiFiClient client = server.available();
    if (client) {
        String req = client.readStringUntil('\r');
        client.readStringUntil('\n');
        
        if (req.indexOf("GET /dashboard/list") != -1) {
            client.println("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"files\": \"" + getFileList() + "\"}");
        } else if (req.indexOf("POST /dashboard/clear") != -1) {
            deleteAllFiles();
            client.println("HTTP/1.1 200 OK\r\n\r\n{\"status\":\"cleared\"}");
        } else {
             client.println("HTTP/1.1 200 OK\r\n\r\nESP32 Rehab Dashboard Ready");
        }
        client.stop();
    }
}

void checkSwitch() {
    static bool lastState = HIGH;
    static unsigned long lastDb = 0;
    bool state = digitalRead(WIFI_SWITCH_PIN);
    
    if (state != lastState) lastDb = millis();
    if ((millis() - lastDb) > 50) {
        if (state == LOW && !wifiMode) {
            wifiMode = true;
            startWiFiHotspot();
        } else if (state == HIGH && wifiMode) {
            wifiMode = false;
            stopWiFi();
        }
    }
    lastState = state;
}

void startWiFiHotspot() {
    WiFi.disconnect(true);
    WiFi.mode(WIFI_AP);
    WiFi.softAP(ap_ssid, ap_password);
    server.begin();
    Serial.println("WiFi Hotspot Active");
}

void stopWiFi() {
    server.stop();
    WiFi.softAPdisconnect(true);
    WiFi.mode(WIFI_OFF);
    Serial.println("WiFi Stopped");
}

String getFileList() {
    String list = "";
    File root = SD.open("/");
    File file = root.openNextFile();
    while(file) {
        if(!file.isDirectory()) {
            if(list.length() > 0) list += ",";
            list += file.name();
        }
        file = root.openNextFile();
    }
    return list;
}

bool deleteAllFiles() {
    SD.remove("/exercise.csv");
    SD.remove("/gait.csv");
    File f = SD.open("/exercise.csv", FILE_WRITE); f.println("Date,Exercise"); f.close();
    f = SD.open("/gait.csv", FILE_WRITE); f.println("Date,Metric,Value"); f.close();
    return true;
}
#include <WiFi.h>
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h> 
#include <utility/imumaths.h>
#include <EloquentTinyML.h> // ML Library
#include "model_data.h"     // Trained model

// CONFIGURATIONS 
#define S3_SDA_PIN 8      
#define S3_SCL_PIN 9      
#define CS_PIN     10            
#define WINDOW_SIZE 100        
#define NUMBER_OF_INPUTS 600   
#define NUMBER_OF_OUTPUTS 4    
#define TENSOR_ARENA_SIZE 60000 

// CALIBRATION VALUES FROM PYTHON OUTPUT 
float MODEL_MEAN[] = { 4.41356022e+00, 4.41356022e+02, 7.35813104e-02, -8.54193334e-01, 2.07999215e-02, 1.71881990e+00}; 
float MODEL_SCALE[] = { 2.92318147e+00, 2.92318147e+02, 3.86334882e-01, 3.58623749e-01, 2.47383006e-01, 2.35500859e+01 }; 

// ML Objects
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);
Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> classifier;

// Globals
float sensorBuffer[WINDOW_SIZE][6]; 
int sampleIndex = 0;                


// SECTION A: SENSOR READING
void getRealTimeSensorData(float* ax, float* ay, float* az, float* gx, float* gy, float* gz) {
    imu::Vector<3> accel = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
    imu::Vector<3> gyro = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);

    *ax = accel.x();
    *ay = accel.y();
    *az = accel.z();

    *gx = gyro.x();
    *gy = gyro.y();
    *gz = gyro.z();
}

// SECTION B: GAIT METRICS 
void calculateGaitMetrics() {
    Serial.println("[METRICS] Calculating Step Count & Sway...");
    // Add specific gait logic here 
    
}

// SECTION C: CLASSIFIER (CNN)
int runModel() {
    float flatInput[WINDOW_SIZE * 6];
    
    for (int i = 0; i < WINDOW_SIZE; i++) {
        for (int j = 0; j < 6; j++) {
            flatInput[(i * 6) + j] = sensorBuffer[i][j];
        }
    }
    
    // Run Prediction
    float prediction[NUMBER_OF_OUTPUTS] = {0};
    classifier.predict(flatInput, prediction);
    
    // Find highest probability
    int bestClass = 0;
    float maxProb = prediction[0];
    
    for (int i = 1; i < NUMBER_OF_OUTPUTS; i++) {
        if (prediction[i] > maxProb) {
            maxProb = prediction[i];
            bestClass = i;
        }
    }
    
    Serial.print("Confidence: "); Serial.println(maxProb);
    
    if (maxProb < 0.60) return 3; // Return Unrecognized if unsure
    
    return bestClass;
}

// SECTION D: LOGGING
void logToSD(String activity) {
    File file = SD.open("/history.csv", FILE_APPEND);
    if(file){
        file.print(millis());
        file.print(",");
        file.println(activity);
        file.close();
    } else {
        Serial.println("ERR: SD Write Failed");
    }
}

// MAIN SETUP
void setup() {
    Serial.begin(115200);
    delay(2000); 

    Wire.begin(S3_SDA_PIN, S3_SCL_PIN);
    // SD Card
    if (!SD.begin(CS_PIN)) {
        Serial.println("SD Card Not Found! (Check CS Pin)");
    } else {
        Serial.println("SD Card Ready");
    }
    // Initialize BNO055
    if (!bno.begin()) {
        Serial.println("BNO055 NOT FOUND! Check Wiring or I2C Address (0x28 or 0x29)");
        while (1) delay(10);
    }
    Serial.println("BNO055 Ready");
    
    bno.setExtCrystalUse(true);

    // Initialize Model
    classifier.begin(model_data);
    Serial.println("CNN Model Loaded Successfully");
}

// MAIN LOOP (50Hz)
void loop() {
    float ax, ay, az, gx, gy, gz;

    getRealTimeSensorData(&ax, &ay, &az, &gx, &gy, &gz);

    // Normalize Data (Math remains the same)
    sensorBuffer[sampleIndex][0] = (ax - MODEL_MEAN[0]) / MODEL_SCALE[0];
    sensorBuffer[sampleIndex][1] = (ay - MODEL_MEAN[1]) / MODEL_SCALE[1];
    sensorBuffer[sampleIndex][2] = (az - MODEL_MEAN[2]) / MODEL_SCALE[2];
    sensorBuffer[sampleIndex][3] = (gx - MODEL_MEAN[3]) / MODEL_SCALE[3];
    sensorBuffer[sampleIndex][4] = (gy - MODEL_MEAN[4]) / MODEL_SCALE[4];
    sensorBuffer[sampleIndex][5] = (gz - MODEL_MEAN[5]) / MODEL_SCALE[5];

    sampleIndex++;

    if (sampleIndex >= WINDOW_SIZE) {
        
        unsigned long startTime = millis();
        int result = runModel();
        unsigned long duration = millis() - startTime;
        
        String status = "";

        if (result == 0) {
            status = "WALKING";
            calculateGaitMetrics();
        } 
        else if (result == 1) status = "SIT_TO_STAND";
        else if (result == 2) status = "SUPINE_TO_SIT";
        else if (result == 3) {
            status = "UNRECOGNIZED";
            Serial.println("Movement unclear.");
        }

        Serial.print("Detected: "); Serial.print(status);
        Serial.print(" (Took "); Serial.print(duration); Serial.println("ms)");
        
        logToSD(status);

        sampleIndex = 0; // Reset Buffer
    }

    delay(20); 
}
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <EloquentTinyML.h>
#include "model_data.h"

// --- SETTINGS ---
#define CS_PIN 5
#define WINDOW_SIZE 100 
#define NUMBER_OF_INPUTS 600 // (WINDOW_SIZE * 6)
#define NUMBER_OF_OUTPUTS 3
#define TENSOR_ARENA_SIZE 32 * 1024
#define K_CONST 0.415   
#define USER_HEIGHT 1.70 

// --- GLOBALS ---
Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml;
float inputBuffer[WINDOW_SIZE][6];
int bufferIndex = 0;

// SD Card Playlist
const char* playlist[] = {"/walk.csv", "/sit.csv", "/supine.csv"};
int currentTrack = 0;
File dataFile;

// --- MATH & METRICS ---
void calculateMetrics() {
  float sumLatSway = 0, meanLat = 0;
  float maxVert = -100, minVert = 100;
  int steps = 0;
  bool peakState = false;

  // 1. Calculate Mean Lateral
  for(int i=0; i<WINDOW_SIZE; i++) meanLat += inputBuffer[i][0];
  meanLat /= WINDOW_SIZE;

  // 2. Calculate Sway & Detect Steps
  for(int i=0; i<WINDOW_SIZE; i++) {
    sumLatSway += pow(inputBuffer[i][0] - meanLat, 2);
    float vert = inputBuffer[i][1];
    
    if (vert > maxVert) maxVert = vert;
    if (vert < minVert) minVert = vert;
    
    // Simple Peak Detection
    if (abs(vert) > 11.0 && !peakState) { steps++; peakState = true; } 
    else if (abs(vert) < 10.0) { peakState = false; }
  }

  float sway = sqrt(sumLatSway / WINDOW_SIZE);
  float stepLen = (steps == 0) ? 0 : (K_CONST * pow((maxVert - minVert), 0.25) * USER_HEIGHT);
  float cadence = (steps / 2.0) * 60.0; 

  // Output Metrics to Serial
  Serial.println(">> ANALYSIS: Walking");
  Serial.print("   Cadence: "); Serial.print(cadence); Serial.println(" steps/min");
  Serial.print("   Step Len: "); Serial.print(stepLen); Serial.println(" m");
  Serial.print("   Sway: "); Serial.println(sway);
  Serial.println("-----------------------------");
}

// --- AI CLASSIFIER ---
void runAI() {
  // Flatten 2D buffer to 1D array
  float flatInput[WINDOW_SIZE * 6];
  for (int i=0; i<WINDOW_SIZE; i++) {
    for (int j=0; j<6; j++) {
      flatInput[(i*6) + j] = inputBuffer[i][j];
    }
  }

  // Predict
  float prediction[3] = {0, 0, 0};
  ml.predict(flatInput, prediction);
  
  // Get Class (Argmax)
  int classIdx = 0;
  if (prediction[1] > prediction[0]) classIdx = 1;
  if (prediction[2] > prediction[classIdx]) classIdx = 2;

  // Handle Result
  if (classIdx == 0) {
    calculateMetrics(); 
  } else if (classIdx == 1) {
    Serial.println(">> DETECTED: Sit-to-Stand");
    Serial.println("-----------------------------");
  } else {
    Serial.println(">> DETECTED: Supine-to-Sit");
    Serial.println("-----------------------------");
  }
}

// --- SD FILE READING ---
bool readLine(float* vals) {
  if (!dataFile) {
    dataFile = SD.open(playlist[currentTrack]);
    if(dataFile) dataFile.readStringUntil('\n'); // Skip header
  }
  
  // Switch track if file ends
  if (!dataFile || !dataFile.available()) {
    if(dataFile) dataFile.close();
    currentTrack = (currentTrack + 1) % 3; 
    Serial.print("\n[INFO] Switching to track: "); Serial.println(playlist[currentTrack]);
    delay(1000); 
    return false;
  }
  
  String line = dataFile.readStringUntil('\n');
  int idx = 0, last = -1;
  float row[8]; // Expecting 8 columns in CSV
  
  for(int i=0; i<line.length(); i++){
    if(line.charAt(i)==','){
      if(idx<8) row[idx] = line.substring(last+1, i).toFloat();
      idx++; last=i;
    }
  }
  if(idx<8) row[idx] = line.substring(last+1).toFloat();
  
  // Map CSV columns to sensor inputs
  vals[0]=row[2]*9.8; vals[1]=row[3]*9.8; vals[2]=row[4]*9.8; 
  vals[3]=row[5];     vals[4]=row[6];     vals[5]=row[7];     
  return true;
}

void setup() {
  Serial.begin(115200);
  while(!Serial);
  Serial.println("Rehabelt System Starting...");

  // Initialize ML
  ml.begin(model_data);
  Serial.println("AI Model Loaded.");

  // Initialize SD
  if (!SD.begin(CS_PIN)) { 
    Serial.println("ERROR: SD Mount Failed!"); 
    while(1); // Stop execution
  }
  Serial.println("SD Card Ready.");
}

void loop() {
  float sensors[6];
  
  // Read one line of data
  if (readLine(sensors)) {
    // Fill Buffer
    for(int i=0; i<6; i++) inputBuffer[bufferIndex][i] = sensors[i];
    bufferIndex++;

    // When buffer is full (100 samples), run AI
    if (bufferIndex >= WINDOW_SIZE) {
      runAI(); 
      bufferIndex = 0; // Reset buffer
    }
    delay(1000); // Simulate sensor sampling rate (1 sec)
  }
}
#include <Arduino.h>
#include "SPIFFS.h"
#include "NeuralNetwork.h"
//#include "eeg_data.h"

NeuralNetwork *nn;

void setup()
{
  // Serial data transmission: bits/sec
  Serial.begin(115200);
  // Neural network instantiation
  nn = new NeuralNetwork();
  // File system
  if(!SPIFFS.begin(true)){
    Serial.println("An Error has occurred while mounting SPIFFS");
    return;
  }
}

void loop()
{
  // Old test with dummy array of 7680 zeroes
  //float dummy_array[7680] = {0};

  // Input & output data size Sanity Check
  Serial.printf("Model running...\n");

  Serial.printf("input->dims->size = %d\n", nn->getInputSize());
  Serial.printf("input->dims->data[0] = %d\n", nn->getInputData()[0]);
  Serial.printf("input->dims->data[1] = %d\n", nn->getInputData()[1]);
  Serial.printf("input->dims->data[2] = %d\n", nn->getInputData()[2]);
  Serial.printf("input->dims->data[3] = %d\n", nn->getInputData()[3]);

  Serial.printf("output->dims->size = %d\n", nn->getOutputSize());
  Serial.printf("output->dims->data[0] = %d\n", nn->getOutputData()[0]);
  Serial.printf("output->dims->data[1] = %d\n", nn->getOutputData()[1]);

  int count = 0;
  for(int j=0;j<=4;j++) {
    unsigned long start_time = millis();

    // Get input data from files
    String filehead = "/mcu_eeg_data0_";
    String fileend = ".txt";
    File file = SPIFFS.open(filehead + j + fileend);

    int label = file.readStringUntil('\n').toInt();
    for(int i = 0; i < 7680; i++) {
      String data = file.readStringUntil('\n');
      nn->getInputBuffer_f()[i] = data.toFloat();
    }

    file.close();

    // Output data buffer
    Serial.printf("output->data.f [");
    nn->invoke();
    float y[5];
    for(int i = 0; i < 5; i++) {
      y[i] = nn->predict_f()[i];
      Serial.printf("%f  ", y[i]);
    }
    Serial.printf("]\n");
    
    // Extract prediction
    int pred_i;
      
    for (int i = 0; i < 5; i++) {
      if (i == 0){
        pred_i = i;
      }
      if (y[i] > y[pred_i]){
        pred_i=i;
      }
    }

    Serial.printf("PREDICTION: CLASS %d\n", pred_i);
    Serial.printf("LABEL: CLASS %d\n", label);
    if(pred_i == label) count++;

    // Timer
    unsigned long end_time = millis();
    Serial.print("Time Usage (ms): ");
    Serial.println(end_time - start_time);
    Serial.print("-----------------------\n");

    /*// Input data buffer
    for(int i = 0; i < 7680; i++) {
      //int8_t x_quantized = sleep_array[i] / nn->getInputScale() + nn->getInputZeroPoint();
      nn->getInputBuffer_f()[i] = sleep_array[i];
    }

    // Output prediction
    nn->invoke();
    int8_t result1_quan = nn->predict()[0];
    int8_t result2_quan = nn->predict()[1];
    int8_t result3_quan = nn->predict()[2];
    int8_t result4_quan = nn->predict()[3];
    int8_t result5_quan = nn->predict()[4];

    // Dequantize the output from integer to floating-point
    float y[5];
    y[0] = (result1_quan - nn->getOutputZeroPoint()) * nn->getOutputScale();
    y[1] = (result2_quan - nn->getOutputZeroPoint()) * nn->getOutputScale();
    y[2] = (result3_quan - nn->getOutputZeroPoint()) * nn->getOutputScale();
    y[3] = (result4_quan - nn->getOutputZeroPoint()) * nn->getOutputScale();
    y[4] = (result5_quan - nn->getOutputZeroPoint()) * nn->getOutputScale();

    Serial.print("-----------------------\n");
    Serial.printf("Input zero point: %d\n", nn->getInputZeroPoint());
    Serial.printf("Input scale: %f\n", nn->getInputScale());
    Serial.printf("Output zero point: %d\n", nn->getOutputZeroPoint());
    Serial.printf("Output scale: %f\n", nn->getOutputScale());
    Serial.printf("Quantized [%d, %d, %d, %d, %d]\n", result1_quan, result2_quan, result3_quan, result4_quan, result5_quan);
    Serial.printf("Result [%f, %f, %f, %f, %f]\n", y[0], y[1], y[2], y[3], y[4]);*/
  }
  Serial.printf("ACCURACY COUNT: %d\n", count);
  
  delay(1000);
}
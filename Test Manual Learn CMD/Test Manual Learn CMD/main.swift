import Foundation
import CoreML

do {
    // Load the compiled model from the bundle
    guard let modelURL = Bundle.main.url(forResource: "DBTCUSD2024", withExtension: "mlmodelc") else {
        fatalError("Model file 'DBTCUSD2024.mlmodelc' not found in the bundle.")
    }
    
    // Load the Core ML model
    let model = try MLModel(contentsOf: modelURL)
    
    // Create input values
    let inputValues: [Float] = [82619, 84082, 82763, 86913, 84204]
    let inputArray = try MLMultiArray(shape: [5], dataType: .float32)
    
    // Populate the multi-array with input values
    for i in 0..<5 {
        inputArray[i] = inputValues[i] as NSNumber
    }
    
    // Create a feature provider with the input named "features"
    let inputFeature = MLFeatureValue(multiArray: inputArray)
    let provider = try MLDictionaryFeatureProvider(dictionary: ["features": inputFeature])
    
    // Make a prediction
    let prediction = try model.prediction(from: provider)
    
    // Extract and validate the output feature value
    guard let outputFeature = prediction.featureValue(for: "target"),
          let outputArray = outputFeature.multiArrayValue else {
        fatalError("Failed to extract output array.")
    }
    
    // Ensure the output array has the expected format
    if outputArray.shape == [1] && outputArray.dataType == .float32 {
        // Extract and print the predicted value
        let predictedValue = outputArray[0].floatValue
        print("Predicted value: \(predictedValue)")
    } else {
        fatalError("Unexpected output format. Expected [Float32], got shape \(outputArray.shape) with type \(outputArray.dataType).")
    }
}
catch {
    print("An error occurred during prediction: \(error.localizedDescription)")
}


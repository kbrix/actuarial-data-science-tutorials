using System.Text.Json;
using Inference.Data;

namespace Inference;

public static class LanguageModelHelper
{
    public static int ExtractVehicleCount(string sentence, HttpClient client)
    {
        const string systemMessage = "Read the text given in the user prompt and answer how many vehicles were involved in the accident. Only output a number.";
        var messages = new Message[]
        {
            new (Role: Role.System, Content: systemMessage),
            new (Role: Role.User, Content: sentence)
        };
        
        var requestBody = new RequestBody(
            Model: Model.Gemma_3_12b, Messages: messages, Temperature: 0.2, MaxTokens: -1, Stream: false);
        
        var jsonSerializerOptions = new JsonSerializerOptions { WriteIndented = true };
        var jsonString = JsonSerializer.Serialize(requestBody, jsonSerializerOptions);
        var content = new StringContent(jsonString, System.Text.Encoding.UTF8, "application/json");
        var response = client.PostAsync("http://localhost:1234/v1/chat/completions", content).Result;

        string responseJson;
        ResponseBody responseBody;
        
        if (response.IsSuccessStatusCode)
        {
            // Deserialize the response JSON
            responseJson = response.Content.ReadAsStringAsync().Result;
            responseBody = JsonSerializer.Deserialize<ResponseBody>(responseJson, jsonSerializerOptions)!;
        }
        else
        {
            throw new Exception($"Error: {response.StatusCode} - {response.Content.ReadAsStringAsync()}");
            // Console.WriteLine($"Error: {response.StatusCode} - {response.Content.ReadAsStringAsync()}");
        }

        var vehicleCountString = responseBody.Choices[0].Message.Content;
        return int.TryParse(vehicleCountString, out var result)
            ? result : throw new Exception($"Could not parse vehicle count: '{vehicleCountString}'...");
    }

    public static void SaveVehicleCountsToDisk(string outputPath, int[] vehicleCounts)
    {
        if (File.Exists(outputPath))
            return;
        
        using var outputFile = new StreamWriter(outputPath);
        foreach (var count in vehicleCounts)
            outputFile.WriteLine(count);
    }
}
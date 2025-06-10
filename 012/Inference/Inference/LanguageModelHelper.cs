using System.Text.Json;
using Inference.Data;

namespace Inference;

public static class LanguageModelHelper
{
    public static int ExtractVehicleCount(string sentence, Model model, HttpClient client)
    {
        const string systemMessage = "Read the text given in the user prompt and answer how many vehicles were involved in the accident. Only output a number. Valid numbers are 1, 2, and 3. Do not output any explanations. Do not output anything else than a single number under ANY circumstances.";
        var messages = new Message[]
        {
            new (Role: Role.System, Content: systemMessage),
            new (Role: Role.User, Content: sentence)
        };
        
        var requestBody = new RequestBody(
            Model: model, Messages: messages, Temperature: 0.2, MaxTokens: -1, Stream: false);
        
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
        var vehicleCount = ExtractContent(vehicleCountString, model);

        return vehicleCount;
    }

    private static int ExtractContent(string content, Model model)
    {
        switch (model)
        {
            case Model.Gemma_3_12b:
            case Model.Gemma_3_27b:
            case Model.IBM_Granite_3_2_28:
            case Model.MS_Phi_4:
            {
                return int.TryParse(content, out var result)
                    ? result : throw new Exception($"Could not parse vehicle count: '{content}'...");
            }

            case Model.Qwen_3_1_7b:
            case Model.Qwen_3_4b:
            case Model.Qwen_3_32b:
            case Model.DeepSeek_R1_Qwen3_8b:
            case Model.MS_Phi_4_Reasoning_Plus:
            {
                var subStringContent = content.Split("</think>");
                return int.TryParse(subStringContent[1], out var result)
                    ? result : throw new Exception($"Could not parse vehicle count: '{subStringContent[1]}'...");
            }

            default:
                throw new ArgumentOutOfRangeException(nameof(model), model, null);
        }
    }

    public static string GetCountResultPath(Model model)
    {
        const string baseOutputPath = "../../../../../counts";

        return model switch
        {
            Model.Gemma_3_12b => baseOutputPath + "-gemma-3-12b.csv",
            Model.Gemma_3_27b => baseOutputPath + "-gemma-3-27b.csv",
            Model.IBM_Granite_3_2_28 => baseOutputPath + "-ibm-granite-3.2-28b.csv",
            Model.Qwen_3_1_7b => baseOutputPath + "-qwen-3-1-7b.csv",
            Model.Qwen_3_4b => baseOutputPath + "-qwen-3-4b.csv",
            Model.Qwen_3_32b => baseOutputPath + "-qwen-3-32b.csv",
            Model.DeepSeek_R1_Qwen3_8b => baseOutputPath + "-deepseek-r1-qwen-3-8b.csv",
            Model.MS_Phi_4 => baseOutputPath + "-ms-phi-4.csv",
            Model.MS_Phi_4_Reasoning_Plus => baseOutputPath + "-ms-phi-4-reasoning-plus.csv",
            _ => throw new ArgumentOutOfRangeException(nameof(model), model, null)
        };
    }

    public static IList<LanguageModelResult> ReadVehicleCountsFromDisk(Model model)
    {
        var results = new List<LanguageModelResult>();
        
        var inputPath = GetCountResultPath(model);
        
        using var inputFile = new StreamReader(inputPath);
        inputFile.ReadLine(); // ignore header
        
        while (!inputFile.EndOfStream)
        {
            var line = inputFile.ReadLine()!;
            var values = line.Split(";");
            var result = new LanguageModelResult(
                RowNumber: int.TryParse(values[0], out var rowNumber) 
                    ? rowNumber : throw new Exception($"Could not parse row number: '{values[0]}'..."),
                TrueCount: int.TryParse(values[1], out var trueCount)
                    ? trueCount : throw new Exception($"Could not parse true count: '{values[1]}'..."),
                PredictedCount: int.TryParse(values[2], out var predictedCount)
                    ? predictedCount : throw new Exception($"Could not parse predicted count: '{values[2]}'..."));
            results.Add(result);
        }
        
        return results;
    }
    
    public static void SaveVehicleCountsToDisk(IList<LanguageModelResult> results, Model model)
    {
        var outputPath = GetCountResultPath(model);
        
        if (File.Exists(outputPath))
            return;

        var headerNames = new[]
        {
            nameof(LanguageModelResult.RowNumber),
            nameof(LanguageModelResult.TrueCount),
            nameof(LanguageModelResult.PredictedCount)
        };
        var headerString = string.Join(";", headerNames);
        
        using var outputFile = new StreamWriter(outputPath);
        outputFile.WriteLine(headerString);

        foreach (var result in results)
        {
            var values = new[]
            {
                result.RowNumber.ToString(),
                result.TrueCount.ToString(),
                result.PredictedCount.ToString()
            };
            var line = string.Join(";", values);
            outputFile.WriteLine(line);
        }
    }
}
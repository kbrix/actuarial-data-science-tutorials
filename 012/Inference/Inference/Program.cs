using System.Diagnostics;
using System.Text.Json;
using Inference.Data;
using Parquet.Serialization;
using BertTokenizer = FastBertTokenizer.BertTokenizer;

namespace Inference;

using Microsoft.ML;
using System;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Hello, World!");

        // Load in data from the parquet file
        const string parquetPath = "../../../../../NHTSA_NMVCCS_extract.parquet.gzip";
        
        var data = ParquetSerializer.DeserializeAsync<ParquetRecord>(parquetPath).Result.ToArray();
        Console.WriteLine($"Parquet data count: {data.Length}.");
        
        // Create the transformer model
        const string modelFile = "../../../../../distilbert-base-multilingual-cased-onnx/model.onnx"; // has output as the last hidden state
        
        var context = new MLContext();
        var useGpu = false;
        
        var estimator = context.Transforms.ApplyOnnxModel(
            modelFile: modelFile,
            inputColumnNames: ["input_ids", "attention_mask"], // Match the input names
            // outputColumnNames: ["logits"], // Match the model output names
            outputColumnNames: ["last_hidden_state"], // Match the model output names
            gpuDeviceId: useGpu ? 0 : null);
        
        var transformer = estimator.Fit(context.Data.LoadFromEnumerable<BertInput>(new List<BertInput>()));
        var predictionEngine = context.Model.CreatePredictionEngine<BertInput, BertOutput>(transformer);
        
        var tokenizer = new BertTokenizer();
        using var vocabularyFile = File.OpenText(@"../../../../../distilbert-base-multilingual-cased-onnx/vocab.txt");
        tokenizer.LoadVocabulary(vocabularyFile, convertInputToLowercase: false);
        
        var limit = data.Length;
        var embeddings = new List<float[]>(limit);
        
        const string embeddingOutputFile = "../../../../../embeddings.csv";
        const string parquetOutputFile = "../../../../../parquet.csv";

        long timestamp;
        TimeSpan elapsedTime;
        
        if (File.Exists(embeddingOutputFile))
        {
            Console.WriteLine($"Reading embedding file: '{embeddingOutputFile}'.");
            embeddings = PredictionHelper.ReadEmbeddingsFromDisk(embeddingOutputFile).ToList();
            Console.WriteLine($"Read embedding file count: {embeddings.Count}.");
        }
        else
        {
            timestamp = Stopwatch.GetTimestamp();
            for (var i = 0; i < limit; i++)
            {
                var pct = 100.0 * (i / (limit - 1.0));
                Console.WriteLine($"{TimeOnly.FromDateTime(DateTime.Now):HH:mm:ss:ffff} - {i:0000} - {pct}%");
                var sentence = data[i].SUMMARY_EN!;
                // Transform the sentence to a float array (an embedding)
                var embedding = PredictionHelper.GenerateSentenceEmbedding(sentence, tokenizer, predictionEngine);
                embeddings.Add(embedding);
            }
            
            elapsedTime = Stopwatch.GetElapsedTime(timestamp);
            Console.WriteLine($"Completed {embeddings.Count} embedding in {elapsedTime:g}.");
            
            Console.WriteLine($"Saving embeddings file '{embeddingOutputFile}'.");
            PredictionHelper.SaveEmbeddingsToDisk(embeddingOutputFile, embeddings);
            Console.WriteLine($"Saved embedding file count: {embeddings.Count}.");
            PredictionHelper.SaveParquetDataToDisk(parquetOutputFile, data);
            Console.WriteLine($"Saved parquet file count: {data.Length}.");
        }
        
        // Classification using logistic regression
        var logisticRegressionContext = new MLContext(seed: 1337);
        
        var dataForLogisticRegression = data.Zip(embeddings, (record, embedding) =>
            new DataPoint { Label = Math.Min(3L, record.NUMTOTV ?? 0).ToString(), Features = embedding }).ToList();
        
        var dataViewForLogisticRegression = logisticRegressionContext.Data.LoadFromEnumerable(dataForLogisticRegression);
        
        var split = logisticRegressionContext.Data.TrainTestSplit(dataViewForLogisticRegression, testFraction: 0.2);
        var trainSet = logisticRegressionContext.Data
            .CreateEnumerable<DataPoint>(split.TrainSet, reuseRowObject: false);
        var testSet = logisticRegressionContext.Data
            .CreateEnumerable<DataPoint>(split.TestSet, reuseRowObject: false);
        
        var pipeline = logisticRegressionContext.Transforms.Conversion
            .MapValueToKey(nameof(DataPoint.Label))
            .Append(logisticRegressionContext.MulticlassClassification.Trainers
                .SdcaMaximumEntropy(labelColumnName: nameof(DataPoint.Label), featureColumnName: nameof(DataPoint.Features)))
            .Append(logisticRegressionContext.Transforms.Conversion
                .MapValueToKey(outputColumnName: nameof(Prediction.PredictedLabel)));
        
        var model = pipeline.Fit(split.TrainSet);
        var transformedTestSet = model.Transform(split.TestSet);
        
        // Inspect the accuracy of the model
        var metrics = logisticRegressionContext.MulticlassClassification.Evaluate(transformedTestSet);
        Console.WriteLine();
        
        Console.WriteLine($"Micro Accuracy: {metrics.MicroAccuracy:F2}"); // 94 % -- decent enough...
        Console.WriteLine($"Macro Accuracy: {metrics.MacroAccuracy:F2}");
        Console.WriteLine($"Log Loss: {metrics.LogLoss:F2}");
        Console.WriteLine($"Log Loss Reduction: {metrics.LogLossReduction:F2}\n");
        
        Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
        
        // Time for the large language model!
        var counts = new int[data.Length];
        var correctCounts = 0;
        
        using var client = new HttpClient { Timeout = TimeSpan.FromMinutes(5) };
        
        timestamp = Stopwatch.GetTimestamp();
        for (int i = 0; i < data.Length; i++)
        // for (int i = 0; i < 100; i++)
        {
            Console.WriteLine($"{TimeOnly.FromDateTime(DateTime.Now):HH:mm:ss:ffff} - {i:0000} / {data.Length - 1}");
            
            var sentence = data[i].SUMMARY_EN!;
            counts[i] = LanguageModelHelper.ExtractVehicleCount(sentence, client);
            
            if (counts[i] == data[i].NUMTOTV)
                correctCounts++;
        }
        elapsedTime = Stopwatch.GetElapsedTime(timestamp);
        Console.WriteLine($"Completed task in {elapsedTime:g}.");
        
        var countOutputPath = "../../../../../counts.txt";
        LanguageModelHelper.SaveVehicleCountsToDisk(countOutputPath, counts);
        
        Console.WriteLine($"Large language model accuracy {correctCounts}/{data.Length} ~ {correctCounts/data.Length:F2}");
    }
}
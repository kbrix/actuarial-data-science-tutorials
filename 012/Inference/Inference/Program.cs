using System.Diagnostics;
using Microsoft.ML;
using Microsoft.ML.Tokenizers;
using Parquet.Serialization;
using BertTokenizer = FastBertTokenizer.BertTokenizer;

namespace Inference;

using BERTTokenizers;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;

public class ParquetRecord
{
    public long? level_0 { get; set; }
    public long? index { get; set; }
    public long? SCASEID { get; set; }
    public string? SUMMARY_EN { get; set; }
    public string? SUMMARY_GE { get; set; }
    public long? INJSEVA { get; set; }
    public long? NUMTOTV { get; set; }
    public long? WEATHER1 { get; set; }
    public long? WEATHER2 { get; set; }
    public long? WEATHER3 { get; set; }
    public long? WEATHER4 { get; set; }
    public long? WEATHER5 { get; set; }
    public long? WEATHER6 { get; set; }
    public long? WEATHER7 { get; set; }
    public long? WEATHER8 { get; set; }
    public long? INJSEVB { get; set; }
}

public class DataPoint
{
    [VectorType(768)]
    public float[] Features { get; set; }

    public string Label { get; set; }
}

public class Prediction
{
    [ColumnName("PredictedLabel")]
    public uint PredictedLabel { get; set; }

    public float[] Score { get; set; }
}

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Hello, World!");

        // Load in data from parquet file
        var parquetPath =
            // @"C:\Users\brix\Documents\DataScience\clone\12 - NLP Using Transformers\NHTSA_NMVCCS_extract.parquet.gzip";
            // @"/Users/brix/Documents/DataScience/clone/12 - NLP Using Transformers/NHTSA_NMVCCS_extract.parquet.gzip";
            @"../../../../../NHTSA_NMVCCS_extract.parquet.gzip";

        var data = ParquetSerializer.DeserializeAsync<ParquetRecord>(parquetPath).Result.ToArray();
        Console.WriteLine($"Parquet data count: {data.Length}.");

        // Create model
        // var modelFile = @"C:\Users\brix\Downloads\distilbert-base-multilingual-cased.onnx";
        var modelFile =
            // @"C:\Users\brix\Downloads\distilbert-base-multilingual-cased-onnx\model.onnx"; // has output as the last hidden state
            // @"/Users/brix/Documents/DataScience/012/distilbert-base-multilingual-cased-onnx/model.onnx"; // has output as the last hidden state
            @"../../../../../distilbert-base-multilingual-cased-onnx/model.onnx"; // has output as the last hidden state

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
        // using var vocabularyFile = File.OpenText(@"C:\Users\brix\Downloads\distilbert-base-multilingual-cased-onnx\vocab.txt");
        // using var vocabularyFile = File.OpenText(@"/Users/brix/Documents/DataScience/012/distilbert-base-multilingual-cased-onnx/vocab.txt");
        using var vocabularyFile = File.OpenText(@"../../../../../distilbert-base-multilingual-cased-onnx/vocab.txt");
        tokenizer.LoadVocabulary(vocabularyFile, convertInputToLowercase: false);

        var limit = data.Length;
        var embeddings = new List<float[]>(limit);

        // var embeddingOutputFile = @"C:\Users\brix\Documents\DataScience\012\embeddings.csv";
        // const string embeddingOutputFile = @"/Users/brix/Documents/DataScience/012/embeddings.csv";
        const string embeddingOutputFile = @"../../../../../embeddings.csv";
        // const string parquetOutputFile = @"/Users/brix/Documents/DataScience/012/parquet.csv";
        const string parquetOutputFile = @"""../../../../../parquet.csv";
        if (File.Exists(embeddingOutputFile))
        {
            Console.WriteLine($"Reading embedding file: '{embeddingOutputFile}'.");
            embeddings = PredictionHelper.ReadEmbeddingsFromDisk(embeddingOutputFile).ToList();
            Console.WriteLine($"Read embedding file count: {embeddings.Count}.");
        }
        else
        {
            var timestamp = Stopwatch.GetTimestamp();
            for (var i = 0; i < limit; i++)
            {
                var pct = 100.0 * (i / (limit - 1.0));
                Console.WriteLine($"{TimeOnly.FromDateTime(DateTime.Now):HH:mm:ss:ffff} - {i:0000} - {pct}%");
                var sentence = data[i].SUMMARY_EN!;
                var embedding = PredictionHelper.GenerateSentenceEmbedding(sentence, tokenizer, predictionEngine);
                embeddings.Add(embedding);
            }
            
            var elapsedTime = Stopwatch.GetElapsedTime(timestamp);
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
            // new DataPoint { Label = record.NUMTOTV.ToString()!, Features = embedding }).ToList();
            new DataPoint { Label = Math.Min(3L, record.NUMTOTV ?? 0).ToString(), Features = embedding }).ToList();
        
        var dataViewForLogisticRegression = logisticRegressionContext.Data.LoadFromEnumerable(dataForLogisticRegression);
        
        var split = logisticRegressionContext.Data.TrainTestSplit(dataViewForLogisticRegression, testFraction: 0.2);
        // var trainSet = logisticRegressionContext.Data
        //     .CreateEnumerable<DataPoint>(split.TrainSet, reuseRowObject: false);
        // var testSet = logisticRegressionContext.Data
        //     .CreateEnumerable<DataPoint>(split.TestSet, reuseRowObject: false);
        
        var pipeline = logisticRegressionContext.Transforms.Conversion
            .MapValueToKey(nameof(DataPoint.Label))
            .Append(logisticRegressionContext.MulticlassClassification.Trainers
                .SdcaMaximumEntropy(labelColumnName: nameof(DataPoint.Label), featureColumnName: nameof(DataPoint.Features)))
            .Append(logisticRegressionContext.Transforms.Conversion
                .MapValueToKey(outputColumnName: nameof(Prediction.PredictedLabel)));
        
        var model = pipeline.Fit(split.TrainSet);
        var transformedTestSet = model.Transform(split.TestSet);
        // var predictions = logisticRegressionContext.Data
        //     .CreateEnumerable<Prediction>(transformedTestSet, reuseRowObject: false).ToList();

        var metrics = logisticRegressionContext.MulticlassClassification.Evaluate(transformedTestSet);
        Console.WriteLine();

        Console.WriteLine($"Micro Accuracy: {metrics.MicroAccuracy:F2}");
        Console.WriteLine($"Macro Accuracy: {metrics.MacroAccuracy:F2}");
        Console.WriteLine($"Log Loss: {metrics.LogLoss:F2}");
        Console.WriteLine($"Log Loss Reduction: {metrics.LogLossReduction:F2}\n");

        Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
    }
}
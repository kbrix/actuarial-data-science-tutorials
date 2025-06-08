using System.Diagnostics;
using System.Globalization;
using FastBertTokenizer;
using Microsoft.ML;

namespace Inference;

public class PredictionHelper
{
    public static float[] GenerateSentenceEmbedding(
        // string sentence, BertBaseTokenizer tokenizer, PredictionEngine<BertInput, BertOutput> predictionEngine)
        string sentence, BertTokenizer tokenizer, PredictionEngine<BertInput, BertOutput> predictionEngine)
    {
        // var encodedSentence = tokenizer.Encode(sentence.Length, sentence);
        var encodedSentence = tokenizer.Encode(sentence);
        // var normalizedEncodedSentence = HelperMethods.Normalize(encodedSentence);
        var (inputIds, attentionMask, _) = HelperMethods.Normalize(encodedSentence);

        var input = new BertInput
        {
            // InputIds = normalizedEncodedSentence.ToArray().Select(x => x.InputIds),
            // AttentionMask = normalizedEncodedSentence.Select(x => x.AttentionMask).ToArray()
            InputIds = inputIds,
            AttentionMask = attentionMask
        };

        var output = predictionEngine.Predict(input); // the output will have length 512 * 768

        var embedding = EmbeddingHelper.ExtractEmbedding(
            output.LastHiddenState, input.AttentionMask, EmbeddingHelper.EmbeddingExtractStrategy.Mean);

        return embedding;
    }

    public static IList<float[]> ReadEmbeddingsFromDisk(string inputPath)
    {
        var embeddings = new List<float[]>(768);

        using var inputFile = new StreamReader(inputPath);
        inputFile.ReadLine(); // ignore header
        
        var nfi = new NumberFormatInfo { NumberDecimalSeparator = "." };
        
        while (!inputFile.EndOfStream)
        {
            var line = inputFile.ReadLine();
            var embedding = line!.Split(";").Select(x => float.Parse(x, nfi)).ToArray();
            embeddings.Add(embedding);
        }
        
        return embeddings;
    }

    public static void SaveParquetDataToDisk(string outputPath, ParquetRecord[] data)
    {
        var valueNames = new[]
        {
            nameof(ParquetRecord.level_0), nameof(ParquetRecord.index), nameof(ParquetRecord.SCASEID), nameof(ParquetRecord.NUMTOTV),
            nameof(ParquetRecord.WEATHER1), nameof(ParquetRecord.WEATHER2), nameof(ParquetRecord.WEATHER3), nameof(ParquetRecord.WEATHER4),
            nameof(ParquetRecord.WEATHER5), nameof(ParquetRecord.WEATHER6), nameof(ParquetRecord.WEATHER7), nameof(ParquetRecord.WEATHER8),
            nameof(ParquetRecord.INJSEVA), nameof(ParquetRecord.INJSEVB)
        };

        var header = string.Join(";", valueNames);
        
        using var outputFile = new StreamWriter(outputPath);
        outputFile.WriteLine(header);
        
        foreach (var record in data)
        {
            var values = new[]
            {
                record.level_0, record.index, record.SCASEID, record.NUMTOTV,
                record.WEATHER1, record.WEATHER2, record.WEATHER3, record.WEATHER4,
                record.WEATHER5, record.WEATHER6, record.WEATHER7, record.WEATHER8,
                record.INJSEVA, record.INJSEVB
            };
            
            var line = string.Join(";", values.Select(x => x.ToString()));
            outputFile.WriteLine(line);
        }
    }
    
    public static void SaveEmbeddingsToDisk(string outputPath, IList<float[]> embeddings)
    {
        if (File.Exists(outputPath))
            return;
        
        foreach (var embedding in embeddings)
            Debug.Assert(embedding.Length == 768);
        
        var header = string.Join(";", Enumerable.Range(1, 768).Select(i => $"x_{i}").ToList());
        
        using var outputFile = new StreamWriter(outputPath);
        outputFile.WriteLine(header);
        
        var nfi = new NumberFormatInfo { NumberDecimalSeparator = "." };

        foreach (var embedding in embeddings)
        {
            var line = string.Join(";", embedding.Select(x => x.ToString(nfi)));
            outputFile.WriteLine(line);
        }
    }
}
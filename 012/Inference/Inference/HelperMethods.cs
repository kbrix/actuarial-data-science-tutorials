using System.Diagnostics;

namespace Inference;

public static class HelperMethods
{
    public static (long[] InputIds, long[] AttentionMask, long[] TokenTypeIds) Normalize(
        (Memory<long> InputIds, Memory<long> AttentionMask, Memory<long> TokenTypeIds) input, int maxLength = 512)
    {
        var inputIds = input.InputIds.ToArray();
        var attentionMask = input.AttentionMask.ToArray();
        var tokenTypeIds = input.TokenTypeIds.ToArray();

        var actualLength = Math.Min(inputIds.Length, maxLength);

        // Truncate to maxLength
        inputIds = inputIds.Take(actualLength).ToArray();
        attentionMask = attentionMask.Take(actualLength).ToArray();
        tokenTypeIds = tokenTypeIds.Take(actualLength).ToArray();

        // Pad if necessary
        if (actualLength < maxLength)
        {
            var padLength = maxLength - actualLength;

            inputIds = inputIds.Concat(Enumerable.Repeat(0L, padLength)).ToArray();
            attentionMask = attentionMask.Concat(Enumerable.Repeat(0L, padLength)).ToArray();
            tokenTypeIds = tokenTypeIds.Concat(Enumerable.Repeat(0L, padLength)).ToArray();
        }

        return (inputIds, attentionMask, tokenTypeIds);
    }
    
    /// <summary>
    /// Truncates or pads the result of encoding a sentence using a BERT tokenizer.
    /// </summary>
    /// <param name="input">The encoded sentence.</param>
    /// <param name="maxLength">The maximum length of the tokens</param>
    /// <returns>Input that can be used in a BERT model.</returns>
    public static List<(long InputIds, long TokenTypeIds, long AttentionMask)> Normalize(
        List<(long InputIds, long TokenTypeIds, long AttentionMask)> input, int maxLength = 512)
    {
        var result = new List<(long InputIds, long TokenTypeIds, long AttentionMask)>(maxLength);

        var count = input.Count;

        if (count > maxLength)
        {
            // Truncate to maxLen
            result = input.Take(maxLength).ToList();
        }
        else if (count < maxLength)
        {
            // Copy original input
            result.AddRange(input);

            // Pad the remaining
            var padLength = maxLength - count;
            var padToken = (InputIds: 0L, TokenTypeIds: 0L, AttentionMask: 0L);

            for (var i = 0; i < padLength; i++)
            {
                result.Add(padToken);
            }
        }
        else
        {
            // Already correct length
            result = new List<(long InputIds, long TokenTypeIds, long AttentionMask)>(input);
        }

        return result;
    }

    
    // protected BertInput BuildInput(List<(string Token, int Index, long SegmentIndex)> tokens)
    // {
    //     // Define the maximum sequence length
    //     int maxLength = 256;
    //
    //     // If the token count is less than 256, we need padding
    //     int paddingCount = Math.Max(0, maxLength - tokens.Count);
    //
    //     // Create padding of 0's if needed
    //     var padding = Enumerable.Repeat(0L, paddingCount).ToList();
    //
    //     // If token count is more than 256, no padding is applied
    //     var tokenIndexes = tokens.Select(token => (long)token.Index).Take(maxLength).Concat(padding).ToArray();
    //     var segmentIndexes = tokens.Select(token => token.SegmentIndex).Take(maxLength).Concat(padding).ToArray();
    //     var inputMask = tokens.Select(o => 1L).Take(maxLength).Concat(padding).ToArray();
    //
    //     return new BertInput()
    //     {
    //         InputIds = tokenIndexes,
    //         AttentionMask = tokens.Select(token => token.)
    //     };
    // }
}
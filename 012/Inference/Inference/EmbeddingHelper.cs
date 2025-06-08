namespace Inference;

public static class EmbeddingHelper
{
    public enum EmbeddingExtractStrategy { First, Mean }

    public static float[] ExtractEmbedding(float[] lastHiddenState, long[] attentionMask, EmbeddingExtractStrategy strategy)
    {
        switch (strategy)
        {
            case EmbeddingExtractStrategy.First:
                return FirstEmbedding(lastHiddenState);
            case EmbeddingExtractStrategy.Mean:
                return MeanEmbedding(lastHiddenState, attentionMask);
            default:
                throw new ArgumentOutOfRangeException(nameof(strategy), strategy, "Unhandled case.");
        }
    }
    
    private static float[] FirstEmbedding(float[] lastHiddenState)
    {
        return lastHiddenState.Take(768).ToArray();
    }

    private static float[] MeanEmbedding(float[] lastHiddenState, long[] attentionMask)
    {
        var meanEmbedding = new float[768];
        var validTokenCount = 0;

        for (var i = 0; i < 512; i++)
        {
            if (attentionMask[i] == 1L)
            {
                validTokenCount++;

                for (var j = 0; j < 768; j++)
                {
                    meanEmbedding[j] += lastHiddenState[i * 768 + j];
                }
            }
        }

        if (validTokenCount > 0)
        {
            for (var j = 0; j < 768; j++)
            {
                meanEmbedding[j] /= validTokenCount;
            }
        }
        else
        {
            throw new Exception("No valid tokens found in the attention mask.");
        }
        
        return meanEmbedding;
    }
}
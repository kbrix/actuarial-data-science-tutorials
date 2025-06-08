using Microsoft.ML.Data;

namespace Inference;

/// For the model 'distilbert-base-multilingual-cased.onnx', see 'https://netron.app' to inspect the model.
public class BertInput
{
    [ColumnName("input_ids")] // Matches 'input_ids' from the model input names
    public required long[] InputIds { get; set; }
    
    [ColumnName("attention_mask")] // Matches 'attention_mask' from the model input names
    public required long[] AttentionMask { get; set; }
}
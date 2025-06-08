using Microsoft.ML.Data;

namespace Inference.Data;

public class DataPoint
{
    [VectorType(768)]
    public float[] Features { get; set; }

    public string Label { get; set; }
}

public class Prediction
{
    [ColumnName("PredictedLabel")]
    public required uint PredictedLabel { get; set; }

    public required float[] Score { get; set; }
}
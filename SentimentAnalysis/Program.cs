using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.Data.DataView;

using SentimentAnalysis.DataStructures;

namespace SentimentAnalysis
{
    class Program
    {
        private static readonly string BaseDatasetsRelativePath = @"../../../../Data";
        private static readonly string DataRelativePath = $"{BaseDatasetsRelativePath}/wikiDetoxAnnotated40kRows.tsv";

        private static string DataPath = GetAbsolutePath(DataRelativePath);

        private static readonly string BaseModelRelativePath = @"../../../../MLModels";
        private static readonly string ModelRelativePath = $"{BaseModelRelativePath}/SentimentModel.zip";

        private static string ModelPath = GetAbsolutePath(ModelRelativePath);

        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 1);

            // BuildTrainEvaluateAndSaveModel(mlContext);
            Common.ConsoleHelper.ConsoleWriteHeader("=============== End of training process ===============");

            while (true)
            {
                Console.WriteLine("Please enter a sentence: ");
                string text = Console.ReadLine();
                TestSinglePredction(mlContext, text);
            }
        }

        private static ITransformer BuildTrainEvaluateAndSaveModel(MLContext mlContext)
        {
            // step 1: Common data loading configuration
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(DataPath, hasHeader: true);

            var testTrainSplit = mlContext.BinaryClassification.TrainTestSplit(dataView, testFraction: 0.2);

            // step 2: Common data process configuration with pipeline data transformations 
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: DefaultColumnNames.Features, inputColumnName: nameof(SentimentIssue.Text));

            // step 3: Set the training algorithm, then create and config the modelBuilder  
            var trainer = mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: DefaultColumnNames.Label, featureColumnName: DefaultColumnNames.Features);
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // step 4: Train the model fitting to the DataSet
            Console.WriteLine("=============== Training the model ===============");
            ITransformer trainedModel = trainingPipeline.Fit(testTrainSplit.TrainSet);

            // step 5: Evaluate the model and show accuracy stats
            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            var predictions = trainedModel.Transform(testTrainSplit.TestSet);

            // step 6: Save/persist the trained model to a .ZIP file

            using (var fs = new FileStream(ModelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(trainedModel, fs);

            Console.WriteLine("The model is saved to {0}", ModelPath);

            return trainedModel;

        }

        // Try/test a single prediction by loding the model from the file, first.
        private static void TestSinglePredction(MLContext mlContext, string text)
        {
            SentimentIssue sampleStatement = new SentimentIssue { Text = text };

            ITransformer trainedModel;
            using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                trainedModel = mlContext.Model.Load(stream);
            }

            // Create prediction engine related to the loaded trained model
            var predEngine = trainedModel.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(mlContext);

            // Score
            var resultprediction = predEngine.Predict(sampleStatement);

            Console.WriteLine($"=============== Single Prediction  ===============");
            Console.WriteLine($"Text: {sampleStatement.Text} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Toxic" : "Non Toxic")} sentiment | Probability: {resultprediction.Probability} ");
            Console.WriteLine($"==================================================");
        }




        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}

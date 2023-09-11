using NeuralNetworkMLP.Entities;
using NeuralNetworkMLP.Services;
using NeuralNetworkMLP.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkMLP
{
    public static class Controller
    {
        public static void Main()
        {
            var dataReading = DataReading.ReadingAndGenerateInputText("../../Files/iris.data");

            FormatIrisData generateFormatted = new FormatIrisData(dataReading);

            List<Sample> sampleFormattedList = generateFormatted.SampleListFormatted;

            Perceptron perceptron = new Perceptron(4, 3, 4, 0.3);

            for (int e = 0; e < 10000; e++)
            {
                double periodTraineer = 0;
                double classificationTraineerError = 0;
                double periodTraineerClassification = 0;

                for (int a = 0; a < sampleFormattedList.Count; a++)
                {
                    double sampleError = 0;
                    double[] inputX = sampleFormattedList[a].CordX;
                    double[] inputY = sampleFormattedList[a].CordY;

                    double[] theta = perceptron.TrainnerExecute(inputX, inputY);

                    for (int i = 0; i < inputY.Length; i++)
                    {
                        sampleError = Math.Abs(inputY[i] - theta[i]);
                        classificationTraineerError = Math.Abs(inputY[i] - ErrorUtil.GetThreshold(theta[i]));

                    }
                    periodTraineer += sampleError;
                    periodTraineerClassification += classificationTraineerError;
                }

                double periodTest = 0;
                double classificationTestError = 0;
                double periodTestClassification = 0;

                for (int a = 0; a < sampleFormattedList.Count; a++)
                {
                    double sampleError = 0;
                    double[] inputX = sampleFormattedList[a].CordX;
                    double[] inputY = sampleFormattedList[a].CordY;

                    double[] theta = perceptron.executar(inputX);

                    for (int i = 0; i < inputY.Length; i++)
                    {
                        sampleError = Math.Abs(inputY[i] - theta[i]);
                        classificationTestError = Math.Abs(inputY[i] - ErrorUtil.GetThreshold(theta[i]));

                    }
                    periodTest += sampleError;
                    periodTestClassification += classificationTestError;
                }

                Console.Write("Epoca: " + e + " - erro treino:" + periodTraineer);
                Console.WriteLine(" - erro treino classification:" + periodTraineerClassification);

                Console.WriteLine("");

                Console.Write("Epoca: " + e + " - erro teste:" + periodTest);
                Console.WriteLine(" - erro teste classification:" + periodTestClassification);
            }

        }
    }
}


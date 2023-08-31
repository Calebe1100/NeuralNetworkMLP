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
                double periodError = 0;
                double classificationError = 0;
                double periodErrorClassification = 0;

                for (int a = 0; a < sampleFormattedList.Count; a++)
                {
                    double sampleError = 0;
                    double[] inputX = sampleFormattedList[a].CordX;
                    double[] inputY = sampleFormattedList[a].CordY;

                    double[] theta = perceptron.TrainnerExecute(inputX, inputY);

                    for (int i = 0; i < inputY.Length; i++)
                    {
                        sampleError = Math.Abs(inputY[i] - theta[i]);
                        classificationError = Math.Abs(inputY[i] - ErrorUtil.GetThreshold(theta[i]));

                    }
                    periodError += sampleError;
                    periodErrorClassification += classificationError;
                }
                Console.Write("Epoca: " + e + " - erro:" + periodError);
                Console.WriteLine(" - erro clssifiction:" + periodErrorClassification);
            }
        }
    }

}

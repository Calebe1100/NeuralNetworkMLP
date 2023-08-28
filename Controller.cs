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
        public static double[][][] Base { get; set; } =
            new double[][][] {
               new double[][] { new double[] { 0, 0 }, new double[] { 0 } },
                new double[][]{ new double[]  { 0 ,1 }, new double[] { 1  } },
                new double[][]{ new double[] { 1 ,0 }, new double[] { 1 } },
                new double[][]{ new double[] { 1 ,1 }, new double[] { 0 } }
            };
        public static void Main()
        {

            Perceptron perceptron = new Perceptron(2, 1, 2, 0.64);

            for (int e = 0; e < 20000; e++)
            {
                double periodError = 0;
                double classificationError = 0;
                double periodErrorClassification = 0;

                for (int a = 0; a < Base.Length; a++)
                {
                    double sampleError = 0;
                    double[] inputX = Base[a][0];
                    double[] inputY = Base[a][1];

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

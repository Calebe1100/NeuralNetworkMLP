using System;
using System.Linq;
using System.Runtime.Serialization;

namespace NeuralNetworkMLP.Services
{
    public class Perceptron
    {
        public int AmountIn { get; set; }
        public int AmountOut { get; set; }
        public int AmountH { get; set; }
        public double[,] NetworksH { get; set; }
        public double[,] NetworksO { get; set; }
        public double Ni { get; set; }

        public Random Random { get; set; } = new Random();
        public Perceptron(int amountIn, int amoutOut,int amountH, double ni)
        {
            this.AmountIn = amountIn;
            this.AmountOut = amoutOut;
            this.AmountH = amountH;
            this.Ni = ni;
            NetworksH = new double[amountIn + 1, amountH];
            NetworksH.Initialize();
            NetworksO = new double[amountH + 1, amountH];
            NetworksO.Initialize();


            this.FillValues();

        }

        public double[] TrainerExecute(double[] input, double[] values)
        {
            double[] formatInput = input.Concat(Enumerable.Repeat(1.0, 1)).ToArray();
            double[] h = new double[AmountH + 1];

            //Calcular a saida da camada H
            for (int j = 0; j < h.Length - 1; j++)
            {
                for (int i = 0; i < formatInput.Length; i++)
                {
                    h[j] += formatInput[i] * NetworksH[i, j];
                }
                h[j] = (1 / (1 + Math.Exp(-h[j])));
            }
            h[h.Length - 1] = 1;

            //Calcular a saida da camada out
            double[] outArray = new double[this.AmountOut];
            for (int j = 0; j < outArray.Length; j++)
            {
                for (int i = 0; i < h.Length; i++)
                {
                    outArray[j] += h[i] * NetworksO[i, j];
                }
                outArray[j] = (1 / (1 + Math.Exp(-outArray[j])));
            }

            //Calcular os deltas da camada out
            double[] deltaO = new double[this.AmountOut];
            for (int j = 0; j < outArray.Length; j++)
            {
                deltaO[j] = outArray[j] * (1 - outArray[j]) * (values[j] - outArray[j]);
            }

            double[] deltaH = new double[this.AmountH];
            double sum;
            for (int i = 0; i < deltaH.Length; i++)
            {
                sum = 0;
                for (int j = 0; j < deltaO.Length; j++)
                {
                    sum += deltaO[j] * NetworksO[i, j];
                }
                deltaH[i] = h[i] * (1 - h[i]) * sum;
            }

            for (int i = 0; i < AmountH; i++)
            {
                for (int j = 0; j < AmountOut; j++)
                {
                    NetworksO[i, j] += Ni * deltaO[j] * h[i];
                }
            }

            for (int i = 0; i < formatInput.Length; i++)
            {
                for (int j = 0; j < AmountH; j++)
                {
                    NetworksH[i, j] += Ni * deltaH[j] * formatInput[i];
                }
            }

            return outArray;
        }


        public double[] TestExecute(double[] xIn)
        {

            xIn = xIn.Concat(Enumerable.Repeat(1.0, 1)).ToArray();

            // Calcula a saída da camada intermediária
            double[] hiddenOut = new double[AmountH + 1]; // representa a saída da camada intermediária

            for (int h = 0; h < AmountH; h++)
            {
                double u = 0;
                for (int i = 0; i < xIn.Length - 1; i++)
                {
                    u += xIn[i] * NetworksH[i,h];
                }
                hiddenOut[h] = Math.Sign(u); 
            }
            hiddenOut[AmountH] = 1;

            // calcula a saida obtida
            double[] teta = new double[xIn.Length - 1];
            for (int j = 0; j < teta.Length; j++)
            {
                double u = 0;
                for (int h = 0; h < hiddenOut.Length; h++)
                {
                    u += hiddenOut[h] * NetworksO[h,j];
                }
                teta[j] = 1 / (1 + Math.Exp(-u));
            }

            return teta;
        }


        private void FillValues()
        {
            double max = 0.3;
            double min = -0.3;

            for (int i = 0; i < this.AmountIn + 1; i++)
            {
                for (int j = 0; j < this.AmountH; j++)
                {
                    this.NetworksH[i, j] = (max * Random.NextDouble()) - min;
                }
            }

            for (int i = 0; i < this.AmountH + 1; i++)
            {
                for (int j = 0; j < this.AmountOut; j++)
                {
                    this.NetworksO[i, j] = (max * Random.NextDouble()) - min;
                }
            }

        }

    }
}

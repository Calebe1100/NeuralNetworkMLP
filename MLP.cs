//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Text;
//using System.Threading.Tasks;

//namespace NeuralNetworkMLP
//{ 
//    public class Mlp
//    {

//        private double[][] wh, wo;
//        private double ni;
//        private int quantIn, quantOut, quantH;

//        public Mlp(int quantIn, int quantOut, int quantIntermN, double ni)
//        {//quantInterN = quantidade interna de neuronios
//         //this.wh = new double[quantIn + 1][quantIntermN];
//            this.wh = matrizAleatoria(quantIn, quantIntermN);
//            //this.wo = new double[quantIntermN + 1][quantOut];
//            this.wo = matrizAleatoria(quantIntermN, quantOut);
//            this.ni = ni;
//            this.quantIn = quantIn;
//            this.quantOut = quantOut;
//            this.quantH = quantIntermN;
//        }

//        public double[] treinar(double[] vetEntrada, double[] y)
//        {
//            double[] vetEntradas = concVet(vetEntrada, new double[] { 1 });
//            double[] h = new double[quantH + 1];
//            h[quantH] = 1;

//            for (int j = 0; j < quantH; j++)
//            {
//                for (int i = 0; i < vetEntradas.length; i++)
//                {
//                    h[j] += vetEntradas[i] * wh[i][j];
//                }
//                h[j] = 1 / (1 + Math.exp(-h[j]));
//            }

//            double o[] = new double[quantOut];
//            for (int j = 0; j < o.length; j++)
//            {
//                for (int i = 0; i < h.length; i++)
//                {
//                    o[j] += h[i] * wo[i][j];
//                }
//                o[j] = 1 / (1 + Math.Exp(-o[j]));
//            }

//            double[] deltaO = new double[quantOut];
//            for (int j = 0; j < o.; j++)
//            {
//                deltaO[j] = o[j] * (1 - o[j]) * (y[j] - o[j]);
//            }

//            double[] deltaH = new double[quantH];
//            double soma;
//            for (int i = 0; i < deltaH.length; i++)
//            {
//                soma = 0;
//                for (int j = 0; j < o.length; j++)
//                {
//                    soma += deltaO[j] * wo[i][j];
//                }
//                deltaH[i] = h[i] * (1 - h[i]) * soma;
//            }

//            //ajuste de pesos da matriz wo
//            for (int i = 0; i < quantH + 1; i++)
//            {
//                for (int j = 0; j < quantOut; j++)
//                {
//                    wo[i][j] = wo[i][j] + this.ni * deltaO[j] * h[i];
//                }
//            }
//            return o;
//        };

//        private double[] concVet(double[] vetUm, double[] vetDois)
//        {
//            int tamanho = vetUm.length + vetDois.length;
//            double[] vetAux = new double[tamanho];

//            System.arraycopy(vetUm, 0, vetAux, 0, vetUm.length);
//            System.arraycopy(vetDois, 0, vetAux, vetUm.length, vetDois.length);

//            return vetAux;
//        };
//        private double[][] matrizAleatoria(int quantIn, int quantOut)
//        {
//            //Random r = new Random();
//            Random r = ThreadLocalRandom.current();
//            double min = -0.03;
//            double max = 0.03;
//            double matriz[][] = new double[quantIn + 1][quantOut];

//            for (int i = 0; i < quantIn + 1; i++)
//            {
//                for (int j = 0; j < quantOut; j++)
//                {
//                    matriz[i][j] = min + (max - min) * r.nextDouble();
//                }
//            }

//            return matriz;
//        };
//    }
//}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkMLP.Utils
{
    public static class ErrorUtil
    {
        public static int GetThreshold(double erro)
        {
            if (erro > 0.5)
            {
                return 1;
            }
            else
            {
                return 0;
            }
        }
    }
}

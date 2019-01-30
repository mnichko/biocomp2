package pl.edu.pg.eti.biocomp.hmm;

public class BaumWelch {
    public BaumWelch(){
    }

    public HMM getHMM(HMM initialHMM, int observations[], int valuesCount, int statesCount, int maxIterations) {
        HMM hmm = initialHMM;
        for (int i = 0; i < maxIterations; ++i){
            double[][] f = hmm.forward(observations);
            double[][] b = hmm.backward(observations);

            double[][] newA = hmm.computeNewA(f, b, observations);
            double[][] newE = hmm.computeNewE(f, b, observations);


            for (int k=0; k < statesCount; ++k) {
                double aSum = 0;
                for (int kNext=0; kNext < statesCount; ++kNext)
                    aSum += newA[k][kNext];

                for (int kNext=0; kNext < statesCount; ++kNext)
                    newA[k][kNext] /= aSum;
            }
            for (int k=0; k < statesCount; ++k) {
                double eSum = 0;
                for (int value=0; value < valuesCount; ++value)
                    eSum += newE[k][value];
                for (int value=0; value < valuesCount; ++value)
                    newE[k][value] /= eSum;
            }
            hmm = new HMM(newA, newE);

        }
        return hmm;
    }
}


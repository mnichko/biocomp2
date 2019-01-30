package pl.edu.pg.eti.biocomp.hmm;

public class Main {

    public static void main(String[] args) {
        double[][] switchingProbabilities = new double[][]{
                {0.8, 0.2},
                {0.2, 0.8}};
        double[][] emissionProbabilities = new double[][]{
                {1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6},
                {0.05, 0.05, 0.05, 0.05, 0.05, 0.75}};

        var observations = new HMM(switchingProbabilities, emissionProbabilities).generate(0, 400);
        var valuesCount = 6;
        var statesCount = 2;
        HMM initialHMM = HMM.makeInitialHMM(valuesCount, statesCount);
        HMM randomHMM = HMM.makeRandomHMM(valuesCount, statesCount);
        var predictedHMM = new BaumWelch().getHMM(initialHMM, observations, valuesCount, statesCount, 300);
        System.out.println(predictedHMM.toString());
    }
}

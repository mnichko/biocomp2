package pl.edu.pg.eti.biocomp.hmm;

import java.util.*;

public class HMM{
    private final Random r;
    private final double[][] switchingProbabilities;
    private final double[][] emissionProbabilities;
    private final int statesCount;
    private final int valuesCount;

    public static HMM makeInitialHMM(int valuesCount, int statesCount) {
        double[][] switchingProbabilities = new double[statesCount][statesCount];
        double[][] emissionProbabilities  = new double[statesCount][valuesCount];
        for (int i = 0; i < statesCount; ++i){
            for(int j = 0; j < statesCount; ++j)
                switchingProbabilities[i][j] = 1.0 / statesCount;
            for(int j = 0; j < valuesCount; ++j)
                emissionProbabilities[i][j] = 1.0 / valuesCount;
        }
        return new HMM(switchingProbabilities, emissionProbabilities);
    }

    public static HMM makeRandomHMM(int valuesCount, int statesCount) {
        Random r = new Random();
        double[][] switchingProbabilities = new double[statesCount][statesCount];
        double[][] emissionProbabilities  = new double[statesCount][valuesCount];
        for (int i = 0; i < statesCount; ++i){
            for(int j = 0; j < statesCount; ++j)
                switchingProbabilities[i][j] = r.nextDouble();
            for(int j = 0; j < valuesCount; ++j)
                emissionProbabilities[i][j] = r.nextDouble();
        }

        for (int k=0; k < statesCount; ++k) {
            double aSum = 0;
            for (int kNext=0; kNext < statesCount; ++kNext)
                aSum += switchingProbabilities[k][kNext];

            for (int kNext=0; kNext < statesCount; ++kNext)
                switchingProbabilities[k][kNext] /= aSum;
        }
        for (int k=0; k < statesCount; ++k) {
            double eSum = 0;
            for (int value=0; value < valuesCount; ++value)
                eSum += emissionProbabilities[k][value];
            for (int value=0; value < valuesCount; ++value)
                emissionProbabilities[k][value] /= eSum;
        }

        return new HMM(switchingProbabilities, emissionProbabilities);
    }

    public HMM(double[][] switchingProbabilities, double[][] emissionProbabilities) {
        r = new Random();  // this could be injected too, but it is here for the ease of implementation

        this.switchingProbabilities = switchingProbabilities;
        this.emissionProbabilities = emissionProbabilities;

        statesCount = emissionProbabilities.length;
        valuesCount = emissionProbabilities[0].length;
    }

    public int[] generate(int initialState, int count){
        var observations = new int[count];
        var currentState = initialState;
        for (int i = 0; i < count; ++i){
            observations[i] = emit(currentState);
            currentState = nextState(currentState);
        }
        return observations;
    }

    private int emit(int currentState){
        var randomProbability = r.nextDouble();
        double probabilitiesSum = 0.0;
        for (var i = 0; i < valuesCount; i++) {

            probabilitiesSum += emissionProbabilities[currentState][i];
            if (probabilitiesSum > randomProbability)
                return i;
        }
        throw new RuntimeException("Emission probabilities do not sum to 1!");
    }


    private int nextState(int currentState){
        var randomProbability = r.nextDouble();
        double probabilitiesSum = 0.0;
        for (var i = 0; i < statesCount; i++) {

            probabilitiesSum += switchingProbabilities[currentState][i];
            if (probabilitiesSum > randomProbability)
                return i;
        }
        throw new RuntimeException("Switching probabilities do not sum to 1!");
    }

    public double[][] forward(int observations[]) {
        double[][] f = new double[observations.length][statesCount];
        // f[i][k] is the probability of reaching state k after i iterations (knowing the outputs)
        // initialize for i==0:
        f[0][0] = 1.0;
        for (int k = 1; k < statesCount; ++k){
            f[0][k] = 0.0;
        }

        for (int i = 1; i < observations.length; ++i) {
            for (int k = 0; k < statesCount; ++k) {
                double stateEntranceProbability = 0.0; // probability of being in state k after i iterations
                for (int kPrev = 0; kPrev < statesCount; ++kPrev)
                    stateEntranceProbability += f[i - 1][kPrev] * switchingProbabilities[kPrev][k];
                f[i][k] = emissionProbabilities[k][observations[i]] * stateEntranceProbability;
            }
        }
        return f;
    }

    public double[][] backward(int observations[]) {
        double[][] b = new double[observations.length][statesCount];
        // initialize for i==L:
        for (int k = 0; k < statesCount; ++k){
            b[observations.length - 1][k] = 1.0;
        }

        for (int i = observations.length - 2; i >= 0; --i) {  // TODO: check index here
            for (int k = 0; k < statesCount; ++k) {
                double stateEntranceProbability = 0.0; // probability of ...
                for (int kNext = 0; kNext < statesCount; ++kNext)
                    stateEntranceProbability += switchingProbabilities[k][kNext]
                            * emissionProbabilities[k][observations[i + 1]] * b[i + 1][kNext];
                b[i][k] = stateEntranceProbability;
            }
        }
        return b;
    }

    @Override
    public String toString(){
        String result = "";
        for (int i = 0; i < statesCount; ++i){
            for (int j = 0; j < statesCount; ++j)
                result += Double.toString(switchingProbabilities[i][j]) + ", ";
            result += "\n";
        }

        result += "\n";
        for (int i = 0; i < statesCount; ++i){
            for (int j = 0; j < valuesCount; ++j)
                result += Double.toString(emissionProbabilities[i][j]) + ", ";
            result += "\n";
        }
        return result;
    }

    private double probability(int state) {
        double probability = 0.0;
        for (int k=0; k < statesCount; ++k){
            probability += switchingProbabilities[k][state];
        }
        return probability;
    }

    public double[][] computeNewE(double[][] f, double[][] b, int[] observations) {
        double[][] newE = new double[statesCount][valuesCount];
        for (int k = 0; k < statesCount; ++k) {
            for (int value = 0; value < valuesCount; ++value) {
                double sum = 0;
                double p = probability(k);
                for (int j = 0; j < observations.length; ++j) {
                    if (observations[j] == value)
                        sum += f[j][k] * b[j][k];  // +1?
                }
                newE[k][value] = sum / p; // /{P
            }
        }
        return newE;
    }


    public double[][] computeNewA(double[][] f, double[][] b, int[] observations){
        double[][] newA = new double[statesCount][statesCount];
        for (int k = 0; k < statesCount; ++k){
            double p = probability(k);
            for (int kNext = 0; kNext < statesCount; ++kNext){
                double sum = 0;
                for (int j = 0; j < observations.length - 1; ++j){
                    for (int i = 0; i < j; ++i){
                        sum += f[i][k]
                                * b[i+1][kNext]
                                * switchingProbabilities[k][kNext]
                                * emissionProbabilities[kNext][observations[i+1]];
                    }
                }
                newA[k][kNext] = sum / p;
            }
        }
        return newA;

    }
}

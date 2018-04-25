package evoClassifier;

import CMAES.CMAEvolutionStrategy;

import java.util.Arrays;

public class LinearOEC {

    private static CMAEvolutionStrategy cmaES = new CMAEvolutionStrategy();
    private double[] initialize;
    private double regularization = 0.0;
    private int numberVariables = 0;
    private int iterations = 0;
    private double [] fitness;
    private double[] w;
    private OptimalMarginResult thresholdResults;
    private MyQuickSort myQuickSort = new MyQuickSort();

    public LinearOEC(int numberVariables, int iterations, double regularization, double[] initialize){
        this.regularization = regularization;
        this.numberVariables = numberVariables;
        this.initialize = new double[numberVariables];
        this.initialize = initialize.clone();
        this.iterations = iterations;
        w = new double[numberVariables];
    }

    public LinearOEC(int numberVariables, double regularization, double[] initialize){
        this.regularization = regularization;
        this.numberVariables = numberVariables;
        this.initialize = new double[numberVariables];
        this.initialize = initialize.clone();
        this.iterations = (int)(Math.ceil(Math.log(numberVariables)*150.0));
        w = new double[numberVariables];
    }

    public LinearOEC(int numberVariables, double regularization){
        this.regularization = regularization;
        this.numberVariables = numberVariables;

        this.numberVariables = numberVariables;
        this.initialize = new double[numberVariables];

        this.iterations = (int)(Math.ceil(Math.log(numberVariables)*150.0));
        w = new double[numberVariables];
    }

    private double[] reorder(double[] a, int[] indices){
        double[] res = new double[a.length];
        for(int i = 0; i < a.length; ++i){
            res[i] = a[indices[i]];
        }
        return res;
    }

    private OptimalMarginResult getOptimalMargin(double[] d, double[] c, double t0, double t1){
        myQuickSort.sort(d);
        double[] ds = myQuickSort.getArray();
        double[] cs = reorder(c, myQuickSort.getIndices());
        int n0l=0;
        int n1l=0;
        double perf = 0.0;
        double coef = 0.0;
        int besti = 0;
        for(int i = 0; i < ds.length - 1; ++i){
            if(cs[i] == 0){n0l ++;}else {n1l++;}
            double acc1 = (n0l/t0)+(1-n1l/t1);
            double acc2 = (n1l/t1)+(1-n0l/t0);
            if(acc1 > perf){
                besti = i;
                coef = -1;
                perf = acc1;
            }
            if(acc2 > perf){
                besti = i;
                coef = 1;
                perf = acc2;
            }
        }
        OptimalMarginResult res = new OptimalMarginResult();
        res.thr = coef * (ds[besti] + ds[besti+1])/2;
        res.coef = coef;
        res.f = (1.0 - perf/2.0);
        res.marg = Math.abs(ds[besti]-ds[besti+1]);
        return res;
    }

    private double sum(double[] a){
        double res = 0.0;
        for(double i : a){
            res += i;
        }
        return res;
    }

    public void fit(double[][] data, double[] classes){
        init();

        double t1 = sum(classes);
        double t0 = (double) classes.length-t1;

        for(int i=0; i < this.iterations; ++i){
//            System.out.println("Iteration: " + i + " " + cmaES.getBestFunctionValue());
            onStep(data, classes, t0, t1);
        }
        this.w = cmaES.getBestX().clone();
        double[] d = dotProductVector(w, data);
        this.thresholdResults = getOptimalMargin(d, classes, t0, t1);
        for(int i =0; i < this.w.length; ++i){
            this.w[i] = this.w[i] * this.thresholdResults.coef;
        }
    }

    public double[] predict(double[][] data){
        double[] v = dotProductVector(this.w, data);
        double[] classes = new double[v.length];

        for (int i = 0; i < v.length; ++i){
            if(v[i]-thresholdResults.thr < 0)
                classes[i] = 1;
            else
                classes[i] = 0;
        }
        return classes;
    }

    private void init(){
        double [] iniStd = new double [this.numberVariables];
        Arrays.fill(iniStd, 1.0);
        fitness = cmaES.init(this.numberVariables, this.initialize, iniStd);
        //cmaES.writeToDefaultFiles(0);
    }
    private double objective(double[] w, double[][] data, double[] c, double t0, double t1){
        double[] d = dotProductVector(w, data);
        OptimalMarginResult res = getOptimalMargin(d, c, t0, t1);

        double f = res.f;
        if(f == 0.0){
            f -= res.marg;
        }
        if(this.regularization > 0.0) {
            double sumW = 0.0;
            for (double v : w) {
                sumW += Math.abs(v);
            }
            f = f + this.regularization * sumW;
        }
        return f;
    }

    private void onStep(double[][] data, double[] c, double t0, double t1){
        double[][] pop = cmaES.samplePopulation(); // get a new population of solutions
        for(int i = 0; i < pop.length; ++i) {    // for each candidate solution i
            fitness[i] = objective(pop[i], data, c, t0, t1); // fitfun.valueOf() is to be minimized
        }
        cmaES.updateDistribution(fitness);         // pass fitness array to update search distribution
        // --- end core iteration step ---
    }
    private double dotProduct(double[] a, double[] b){
        double res = 0.0;
        for (int i = 0; i < a.length; ++i){
            res += a[i]*b[i];
        }
        return res;
    }
    private double[] dotProductVector(double[] a, double[][] b){
        double[] res = new double[b.length];

        for (int j = 0; j < b.length; ++j) {
            res[j] = dotProduct(a, b[j]);
        }

        return res;
    }
}

class OptimalMarginResult{
    public double thr;
    public double coef;
    public double f;
    public double marg;
    public OptimalMarginResult(){}

}

class MyQuickSort {

    private double array[];
    private int[] indices;

    public void sort(double[] inputArr) {

        if (inputArr == null || inputArr.length == 0) {
            return;
        }
        this.array = inputArr;
        indices = new int[array.length];

        for(int i = 0; i < array.length; ++i){
            indices[i] = i;
        }

        quickSort(0, array.length - 1);
    }

    public int[] getIndices(){
        return indices;
    }

    public double[] getArray(){
        return array;
    }

    private void quickSort(int lowerIndex, int higherIndex) {
        int i = lowerIndex;
        int j = higherIndex;
        // calculate pivot number, I am taking pivot as middle index number
        double pivot = array[lowerIndex+(higherIndex-lowerIndex)/2];
        // Divide into two arrays
        while (i <= j) {
            while (array[i] < pivot) {
                i++;
            }
            while (array[j] > pivot) {
                j--;
            }
            if (i <= j) {
                exchangeNumbers(i, j);
                //move index to next position on both sides
                i++;
                j--;
            }
        }
        // call quickSort() method recursively
        if (lowerIndex < j)
            quickSort(lowerIndex, j);
        if (i < higherIndex)
            quickSort(i, higherIndex);
    }

    private void exchangeNumbers(int i, int j) {
        double temp = array[i];
        array[i] = array[j];
        array[j] = temp;

        int tempInd = indices[i];
        indices[i] = indices[j];
        indices[j] = tempInd;
    }
}
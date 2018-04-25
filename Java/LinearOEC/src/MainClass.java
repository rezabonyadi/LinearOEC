import evoClassifier.LinearOEC;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.LinkedList;

public class MainClass {
    public static void main(String[] args){
//        String csvFile = "../../Data/BC.csv";
        String csvFile = "../../Data/HV.csv";

        LinkedList<double[]> data = new LinkedList<>();
        LinkedList<Double> classes = new LinkedList<>();
        getData(csvFile, data, classes);
        double[][] arrayData = new double[data.size()][data.get(0).length];
        double[] arrayClasses = new double[classes.size()];

        for(int i = 0; i < data.size(); ++i){
            arrayData[i] = data.get(i);
            arrayClasses[i] = classes.get(i);
        }

        long startTime = System.nanoTime();

        // Building and fitting the linear model
        LinearOEC oec = new LinearOEC(arrayData[0].length, 0.0);
        oec.fit(arrayData, arrayClasses);

        long endTime   = System.nanoTime();
        long totalTime = endTime - startTime;

        System.out.println("Optimization time: " + totalTime/1000000 + " (ms)");

        // Evaluating the model for the data (here test and train data are the same for this example)
        double[] estimatedClasses = oec.predict(arrayData);
        double loss = 0.0;
        for(int i = 0; i < estimatedClasses.length; ++i){
            if(Math.abs(estimatedClasses[i] - arrayClasses[i]) > 0){
                loss++;
            }
        }
        System.out.print("Total loss: " + loss);
    }

    static void getData(String csvFile, LinkedList<double[]> data, LinkedList<Double> classes){
        String line = "";
        String cvsSplitBy = ",";

        try {
            BufferedReader br = new BufferedReader(new FileReader(csvFile));
            line = br.readLine();
            while ((line = br.readLine()) != null) {

                // use comma as separator
                String[] country = line.split(cvsSplitBy);
                double[] d = new double[country.length - 1];

                for(int i = 0; i < country.length - 1; ++i){
                    d[i] = Double.parseDouble(country[i]);
                }
                classes.add(Double.parseDouble(country[country.length - 1]));
                data.add(d);
            }
        }catch (Exception e) {
            e.printStackTrace();
        }


    }

}

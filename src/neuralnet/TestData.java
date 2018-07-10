package neuralnet;

import java.io.*;
import java.util.*;

/**
 * @author Alexis Varsava <av11sl@brocku.ca>
 * @version 2
 * @since 2015-11-29
 *
 * Stores and randomizes the data set.
 */
public class TestData {

    private ArrayList<ArrayList<Double>> data;
    //getXSize() is # of rows
    //getYSize() is # of cols
    //I know, I know.
    private int xsize, ysize, outputColumn, outputNodes;
    private int[] inputColumns;
    private RNG generator;
    private static BufferedReader inputReader, fileReader;
    private static HashMap dict;

    /**
     * Constructor. Fills in data then shuffles the order.
     *
     * @param gen Random Number Generator.
     */
    public TestData(RNG gen) {
        generator = gen;
        inputReader = new BufferedReader(new InputStreamReader(System.in));
        dict = new HashMap();
        setFileReader();
        data = populateData();
        xsize = getXSize();
        ysize = getYSize();
        setAttributes();
        outputColumn = setOutputColumn();
        outputNodes = setClassifications();
        scramble();
    }

    /**
     *
     * @return # of input neurons there should be
     */
    public int getAttributes() {
        return inputColumns.length;
    }

    private void setAttributes() {
        String temp = null;
        String[] cols = null;

        System.out.println("We've detected " + getYSize() + " columns.");
        System.out.println("Please enter which to use as input," 
                + "comma-separated.");
        try {
            System.out.print("> ");
            temp = inputReader.readLine();
            cols = temp.split(",");
        } catch (Exception e) {
            System.err.println("Something went wrong in setAttributes.");
        }

        inputColumns = new int[cols.length];

        for (int i = 0; i < cols.length; i++) {
            inputColumns[i] = Integer.parseInt(cols[i]);
        }
    }

    public int getYSize() {
        return data.get(0).size();
    }

    public int getXSize() {
        return data.size();
    }

    public ArrayList<ArrayList<Double>> getData() {
        return data;
    }

    /**
     *
     * @return # of output neurons there should be
     */
    public int getClassifications() {
        return outputNodes;
    }

    private int setClassifications() {
        HashSet uniques = new HashSet(); //does not allow duplicates
        for (int i = 0; i < getXSize(); i++) {
            //will discard if value already exists
            uniques.add(data.get(i).get(getOutputColumn()));
        }
        return uniques.size();
    }

    public int getOutputColumn() {
        return outputColumn - 1;
    }

    private int setOutputColumn() {
        int temp = 0;

        System.out.println("We've detected " + getYSize() + " columns.");
        System.out.println("Please enter which to use as output.");
        try {
            System.out.print("> ");
            temp = Integer.parseInt(inputReader.readLine());
        } catch (Exception e) {
            System.err.println("Something went wrong in setAttributes.");
        }
        return temp;
    }

    private void setFileReader() {
        String path = null;
        File f = null;

        System.out.println("Please enter the path of the data set.");
        while (true) {
            try {
                System.out.print("> ");
                path = inputReader.readLine();
                f = new File(path);
                fileReader = new BufferedReader(new FileReader(f));
                fileReader.mark((int) f.length());
                break;
            } catch (FileNotFoundException fnf) {
                System.err.println("Error: File not found");
            } catch (NullPointerException np) {
                System.err.println("Error: null file");
            } catch (Exception e) {
                System.err.println("Error reading filepath.");
            }
        }
    }

    public double getErrorMargin(InputNeuron[] InputNeurons, double value,
            int placement) {
        double[] ins = new double[getXSize() - 1];
        double result = 0.0;

        //Gather all the input values for comparison
        for (int i = 0; i < InputNeurons.length; i++) {
            ins[i] = InputNeurons[i].getValue();
        }

        result = arraySearch(ins);
        
        if (placement == (int) result) {
            return 1.0 - value;
        }else{
            return 0 - value;
        }
    }

    /**
     *
     * @param InputNeurons
     * @param value What the output neuron ended up firing
     * @param placement Which neuron did the firing
     * @return
     */
    public boolean isCorrect(InputNeuron[] InputNeurons, double value,
            int placement) {
        double[] ins = new double[InputNeurons.length];

        /**
         * What value should have been fired
         */
        double result = 0.0;

        //Gather all the input values for comparison
        for (int i = 0; i < InputNeurons.length; i++) {
            ins[i] = InputNeurons[i].getValue();
        }

        result = arraySearch(ins);
        
        if(placement == result) {
            return Math.round(value) == 1;
        }else{
            return Math.round(value) == 0;
        }
    }

    /**
     * Finds & returns the last value in the array matching the provided one
     *
     * Assumes the last column is always the one you're looking for, oops
     *
     * @param head the first values in the array
     * @return the last value in the matching row
     */
    public double arraySearch(double[] head) {
        for (int i = 0; i < getXSize(); i++) {
            for (int j = 0; j < getYSize() - 1; j++) {
                if (data.get(i).get(j) != head[j]) {
                    break;
                }
                if (j == getYSize() - 2) {
                    return data.get(i).get(getYSize() - 1);
                }
            }
        }
        return Double.NaN;
    }

    public void scramble() {
        //Fisher-Yates Shuffle
        ArrayList<Double> temp = new ArrayList<>();
        for (int i = 0; i < getXSize() - 1; i++) {
            int r = generator.getIntInRange(i, getXSize() - 1);

            temp = new ArrayList<>(data.get(i));
            data.set(i, new ArrayList<>(data.get(r)));
            data.set(r, new ArrayList<>(temp));
        }
    }

    private ArrayList<ArrayList<Double>> populateData() {
        String[] str;
        ArrayList<ArrayList<Double>> dat = new ArrayList<>();
        ArrayList<Double> entry;
        double val;
        int row = 0, v = 0;

        //Put everything in the matrix
        while (true) {
            //make a new row
            entry = new ArrayList<>();
            try {
                //read in a new line and split it by commas
                str = fileReader.readLine().split(",");
            } catch (Exception ex) {
                break;
            }

            //For every item in the row
            for (int i = 0; i < str.length; i++) {
                //if dealing with the expected output column
                if (i == outputColumn) {
                    //if it's not in the dictionary, add it
                    if (!dict.containsKey(str[i])) {
                        //find the highest value in the dictionary
                        while (dict.containsValue(v)) {
                            v++;
                        }
                        //add a new association to the dictionary
                        dict.put(str[i], v);
                    }
                    //add the dictionary value rather than the actual
                    //this is some Java witchcraft, but it works
                    entry.add((double) (int)dict.get(str[i]));
                    
                } else {//if we're dealing with a normal column
                    try {
                        //add it to the data
                        val = Double.parseDouble(str[i]);
                        entry.add(val);
                    } catch (NumberFormatException nfe) {
                        //if it's not a valid number, just blank it out
                        entry.add(0.0);
                    }
                }
            }
            //add the assembled row to our matrix
            dat.add(row, entry);
            row++;
        }

        return dat;
    }
}

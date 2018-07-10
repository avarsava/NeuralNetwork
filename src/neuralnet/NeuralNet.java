package neuralnet;

import java.io.*;

/**
 * @author Alexis Varsava <av11sl@brocku.ca>
 * @version 2
 * @since 2015-11-25
 *
 * Creates and trains an Artificial Neural Network for a user-defined goal.
 * Currently assumes only 1 layer of hidden nodes.
 * 
 * This assignment is, admittedly, a trainwreck. Included the 
 * 4-bit parity data set because it's all I could get to work.
 */
public class NeuralNet {

    final double MIN_RANGE, MAX_RANGE;
    
    int numHidNeurons;
    double alpha;
    Weight[][] InputToHidden, HiddenToOutput;
    InputNeuron[] InputNeurons;
    CalcNeuron[] HiddenNeurons, OutputNeurons;
    RNG generator;
    static BufferedReader reader;
    TestData dat;
    Propagation prop;

    /**
     * Constructor. Assigns values to constants and defines objects. Then,
     * launches program logic.
     */
    public NeuralNet() {
        MIN_RANGE = -1;
        MAX_RANGE = 1;
        
        reader = new BufferedReader(new InputStreamReader(System.in));
        generator = new RNG(MIN_RANGE, MAX_RANGE);
        dat = new TestData(generator);
        
        numHidNeurons = setNumHidden();
        alpha = setAlpha();
        
        prop = chooseProp();
        InputNeurons = new InputNeuron[dat.getAttributes()];
        HiddenNeurons = new CalcNeuron[numHidNeurons];
        OutputNeurons = new CalcNeuron[dat.getClassifications()];
        InputToHidden = setUpWAM(dat.getAttributes(), numHidNeurons);
        HiddenToOutput = setUpWAM(numHidNeurons, dat.getClassifications());
        InputNeuron.initNeurons(InputNeurons);
        setUpHidden();
        inputConnection(HiddenNeurons, InputNeurons);
        inputConnection(OutputNeurons, HiddenNeurons);
        outputConnection(HiddenNeurons, OutputNeurons);
        
        run();
    }

    public TestData getTestData() {
        return dat;
    }

    public InputNeuron[] getInputNeurons() {
        return InputNeurons;
    }

    public CalcNeuron[] getHiddenNeurons() {
        return HiddenNeurons;
    }

    public CalcNeuron[] getOutputNeurons() {
        return OutputNeurons;
    }

    public int getNumHidden() {
        return numHidNeurons;
    }

    public double getAlpha() {
        return alpha;
    }
    
    public Weight[][] getItoH(){
        return InputToHidden;
    }
    
    public Weight[][] getHtoO(){
        return HiddenToOutput;
    }
    
    private double setAlpha(){
        double num = 0;
        do{
            System.out.println("Please enter the learning rate.");
            System.out.println("Recommended value is 0.1.");
            try{
                System.out.print("> ");
                num = Double.parseDouble(reader.readLine());
            }catch(Exception e){
                System.err.println("Please a number greater than 0.");
            }
        }while(num<=0);
        return num;
    }
    
    private int setNumHidden(){
        int num = 0;
        do{
            System.out.println("Please enter the number of hidden neurons.");
            try{
                System.out.print("> ");
                num = Integer.parseInt(reader.readLine());
            }catch(Exception e){
                System.err.println("Please a number greater than 0.");
            }
        }while(num<1);
        return num;
    }
    
    private Propagation chooseProp(){
        int choice = 0;
        do {
            System.out.println("Please select a propagation technique:");
            try {
                System.out.print("(1) Backprop, (2) Delta-Bar-Delta > ");
                choice = Integer.parseInt(reader.readLine());
            } catch (Exception ex) {
                System.err.println("Please enter 1 or 2!");
            }
        } while (choice != 1 && choice != 2);
        if(choice == 1){
            return new BackProp();
        }else{
            return new DeltaBarDelta();
        }
    }

    /**
     * Connects Sigmoid Neurons to the preceding layer.
     *
     * @param neurons a layer of sigmoid (hidden or output) neurons
     * @param inputs the preceding layer. May be input or sigmoid neurons.
     */
    public void inputConnection(CalcNeuron[] neurons, Neuron[] inputs) {
        for (int i = 0; i < neurons.length; i++) {
            neurons[i].setInputs(inputs);
        }
    }

    /**
     * Connects Sigmoid Neurons to the following layer.
     *
     * This is only required for calc neurons because input neurons will
     * never need to calculate the amount of error they are responsible for.
     *
     * @param neurons a layer of sigmoid neurons
     * @param outputs the following layer of sigmoid neurons
     */
    public void outputConnection(CalcNeuron[] neurons, CalcNeuron[] outputs) {
        for (int i = 0; i < neurons.length; i++) {
            neurons[i].setOutputs(outputs);
        }
    }

    /**
     * Assigns random values (but never 0) to every connection in a Weighted
     * Adjacency Matrix.
     *
     * @param xSize size of x axis (left side of neural net)
     * @param ySize size of y axis (right side of neural net)
     * @return populated Weighted Adjacency Matrix
     */
    public Weight[][] setUpWAM(int xSize, int ySize) {
        Weight[][] wam = new Weight[xSize][ySize];

        for (int i = 0; i < xSize; i++) {
            for (int j = 0; j < ySize; j++) {
                wam[i][j] = new Weight(generator.getRandomNotZero(), alpha);
            }
        }
        return wam;
    }

    /**
     * Lets the user choose between the logistic and tanh activation functions
     */
    public void setUpHidden() {
        int choice = 0;
        do {
            System.out.println("Please select an activation function:");
            try {
                System.out.print("(1) Sigmoid, (2) Tanh > ");
                choice = Integer.parseInt(reader.readLine());
            } catch (Exception ex) {
                System.err.println("Please enter 1 or 2!");
            }
        } while (choice != 1 && choice != 2);

        if (choice == 1) {
            SigmoidNeuron.initNeurons(HiddenNeurons, InputToHidden, 
                    HiddenToOutput);
            SigmoidNeuron.initNeurons(OutputNeurons, HiddenToOutput);
        } else {
            TanhNeuron.initNeurons(HiddenNeurons, InputToHidden, 
                    HiddenToOutput);
            TanhNeuron.initNeurons(OutputNeurons, HiddenToOutput);
        }
    }

    /**
     * Prints out "both" weighted adjacency matrices.
     *
     * The quantity of matrices has been baked in for convenience.
     */
    public void printInfo() {
        System.out.println();
        System.out.println("Here's the input-hidden adjacency matrix:");
        printWAM(InputToHidden, dat.getAttributes(), numHidNeurons);
        System.out.println();
        System.out.println("Here's the hidden-output adjacency matrix:");
        printWAM(HiddenToOutput, numHidNeurons, dat.getClassifications());
        System.out.println();
    }

    /**
     * Prints out every value in a weighted adjacency matrix.
     *
     * @param wam matrix to print
     * @param xSize size of x axis
     * @param ySize size of y axis
     */
    public void printWAM(Weight[][] wam, int xSize, int ySize) {
        for (int i = 0; i < ySize; i++) {
            for (int j = 0; j < xSize; j++) {
                System.out.print(j + "," + i + ": " + wam[j][i] + " ");
            }
            System.out.println();
        }
    }

    /**
     * Run loop. Offers the choice between training the ANN, inputting a nybble
     * by hand to test with, or quitting the program.
     */
    public void run() {
        boolean run = true;
        String s = "";
        int option = 0;
        do {
            printInfo();
            do {
                System.out.println("What would you like to do?");
                System.out.println("{1) Training (2) Test (3) Quit");
                System.out.print("> ");
                try {
                    s = reader.readLine();
                } catch (IOException ex) {
                    System.err.println("Failed to read input.");
                }

                try {
                    option = Integer.parseInt(s);
                } catch (Exception e) {
                    System.err.println("Please enter 1, 2, or 3.");
                }
            } while (option != 1 && option != 2 && option != 3);

            switch (option) {
                case 1:
                    prop.doTraining(this);
                    break;
                case 2:
                    doTest();
                    break;
                case 3:
                    run = false;
                    break;
            }
        } while (run == true);
    }

    /**
     * Gets a nybble from the user then runs the ANN on it.
     *
     * Reports back on the result the ANN calculated and whether it was right.
     */
    public void doTest() {
        double[] result = new double[dat.getClassifications()];

        getInput();
        for (int i = 0; i < dat.getClassifications(); i++) {
            result[i] = OutputNeurons[i].getValue();
            System.out.print("Output " + i + ": " + result[i]);
            if(dat.isCorrect(InputNeurons, result[i], i)){
                System.out.println(" Good!");
            }else{
                System.out.println(" Bad.");
            }
        }
    }

    /**
     * Gets data from user, one bit at a time.
     */
    public void getInput() {
        String s = "";
        int n = 2;
        System.out.println("Currently accepting " + dat.getAttributes() 
                + " bits.");
        for (int i = 0; i < dat.getAttributes(); i++) {
            do {
                System.out.print(i + "> ");
                try {
                    s = reader.readLine();
                } catch (Exception e) {
                    System.err.println("Failed to read input");
                }
                try {
                    n = Integer.parseInt(s);
                } catch (Exception e) {
                    System.err.println("Please enter 0 or 1.");
                }
                if (n != 0 && n != 1) {
                    System.err.println("Please enter 0 or 1.");
                }
            } while (n != 0 && n != 1);

            InputNeurons[i].setValue((double) n);
        }
    }

    /**
     * Main method. Entry and exit point of program.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        NeuralNet n = new NeuralNet();
    }

}

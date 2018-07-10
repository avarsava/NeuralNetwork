package neuralnet;

/**
 * @author Alexis Varsava <av11sl@brocku.ca>
 * @version 1
 * @since 2015-11-26
 * 
 * Describes a neuron which holds pure input, does not use a function to 
 * manipulate that data.
 */

public class InputNeuron implements Neuron{
    private double value;
    private int id;
    
    /**
     * Constructor.
     * 
     * @param n ID by which to uniquely identify neuron. Used for weighted 
     * adjacency matrix lookup.
     */
    public InputNeuron(int n){
        id = n;
    }
    
    /**
     * Getter for value.
     * 
     * @return value, gotten from training data or user
     */
    @Override
    public double getValue(){
        return value;
    }
    
    /**
     * Another getter for value.
     * 
     * The repetition is only to make updateWeights() in SigmoidNeuron easier
     * to write.
     * 
     * @return value, gotten from training data or user
     */
    @Override
    public double getSavedValue(){
        return value;
    }
    
    /**
     * Setter for value.
     * 
     * @param n value which to set value.
     */
    public void setValue(double n){
        value = n;
    }
    
    /**
     * Getter for ID
     * 
     * @return ID
     */
    @Override
    public int getID(){
        return id;
    }
    
    /**
     * Initialize input neurons with unique IDs.
     * 
     * @param neurons list of neurons to initialize.
     */
    public static void initNeurons(Neuron[] neurons){
        for (int i = 0; i < neurons.length; i++){
            neurons[i] = new InputNeuron(i);
        }
    }

    /**
     * This is only here to make accepting a Neuron easier elsewhere.
     * 
     * @return nothing
     */
    @Override
    public double getErr() {
        throw new UnsupportedOperationException("getErr InputNeuron"); 
    }
}

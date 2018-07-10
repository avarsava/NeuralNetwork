package neuralnet;

/**
 * @author Alexis Varsava <av11sl@brocku.ca>
 * @version 2
 * @since 2015-11-26
 * 
 * Interface to bring together InputNeurons and SigmoidNeurons.
 */
public interface Neuron {

    /**
     * Every neuron will have an ID unique within its layer.
     * 
     * Set here to a dummy value.
     */
    int id = Integer.MAX_VALUE;

    /**
     * Getter for value
     * 
     * @return value
     */
    public double getValue();

    /**
     * Getter for previously-saved value.
     * 
     * In case one needs to not re-calculate the value.
     * 
     * @return stored value
     */
    public double getSavedValue();

    /**
     * Getter for ID. 
     * 
     * @return ID
     */
    public int getID();

    /**
     * Getter for error
     * 
     * @return error
     */
    public double getErr();
    
    
    
    
}

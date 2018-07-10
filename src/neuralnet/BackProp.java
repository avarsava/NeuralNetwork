package neuralnet;

/**
 * @author Alexis Varsava <av11sl@brocku.ca>
 * @version 1
 * @since 2016-02-19
 * 
 * Classic back-propagation
 */

public class BackProp implements Propagation{
    
    @Override
    public void doTraining(NeuralNet nn){
        double[] result = new double[nn.getTestData().getClassifications()];
        double correct = 0, tempCorrect, epoch = 1, sr = 0;

        while(epoch < 2500) {
            correct = 0;
            for (int x = 0; x < (nn.getTestData().getXSize()); x++) {
                tempCorrect = 0;
                
                //load input
                for (int y = 0; y < nn.getTestData().getYSize()-1; y++) {
                    nn.getInputNeurons()[y].setValue(nn.getTestData()
                            .getData().get(x).get(y));
                }
                
                //get results
                for(int i = 0; i < nn.getTestData().getClassifications(); i++){
                    result[i] = nn.getOutputNeurons()[i].getValue();
                }

                //set error for output nodes
                for (int i = 0; i < nn.getTestData().getClassifications(); i++) 
                {
                    if(nn.getTestData().isCorrect(nn.getInputNeurons(), 
                            result[i], i)){
                        tempCorrect++;
                    }
                    double error = nn.getTestData().getErrorMargin(
                            nn.getInputNeurons(), result[i], i);
                    nn.getOutputNeurons()[i].setErr(error);
                }
                if (tempCorrect == nn.getTestData().getClassifications()){
                    correct++;
                }

                //set error for hidden nodes
                for (int i = 0; i < nn.getNumHidden(); i++) {
                    nn.getHiddenNeurons()[i].setErr(nn.getHiddenNeurons()[i]
                            .calcErr());
                }

                //update weights for hidden nodes
                for (int i = 0; i < nn.getNumHidden(); i++) {
                    nn.getHiddenNeurons()[i].updateWeights(this);
                }

                //update weights for output nodes
                for (int i = 0; i < nn.getTestData().getClassifications(); i++)
                {
                    nn.getOutputNeurons()[i].updateWeights(this);
                }
            }
            sr = correct/(nn.getTestData().getXSize());
            System.out.println("Epoch: " + epoch + ", Success rate: " + sr);
            if(sr == 1.0) break;
            epoch++;
            
            nn.getTestData().scramble();
        }
        System.out.println("Trained with " + epoch * nn.getTestData()
                .getXSize() + " examples!");
    }
    
    
}

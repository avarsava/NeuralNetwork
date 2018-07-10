package neuralnet;

/**
 * @author Alexis Varsava <av11sl@brocku.ca>
 * @version 1
 * @since 2016-03-02
 *
 * Implements Delta-Bar-Delta as it is described in the lecture slides 
 */
public class DeltaBarDelta implements Propagation {
    final double D, K;
    
    public DeltaBarDelta(){
        D = 0.2;
        K = 0.02;
    }

    @Override
    public void doTraining(NeuralNet nn) {
        double[] result = new double[nn.getTestData().getClassifications()];
        double correct = 0, tempCorrect, epoch = 1, sr = 0;
        int numWeights;

        while (epoch < 2500) {
            correct = 0;
            for (int x = 0; x < nn.getTestData().getXSize(); x++) {
                tempCorrect = 0;

                //load input
                for (int y = 0; y < nn.getTestData().getYSize() - 1; y++) {
                    nn.getInputNeurons()[y].setValue(nn.getTestData()
                            .getData().get(x).get(y));
                }

                //get results
                for (int i = 0; i < nn.getTestData().getClassifications(); i++)
                {
                    result[i] = nn.getOutputNeurons()[i].getValue();
                }

                //set error for output nodes
                for (int i = 0; i < nn.getTestData().getClassifications(); i++) 
                {
                    if (nn.getTestData().isCorrect(nn.getInputNeurons(),
                            result[i], i)) {
                        tempCorrect++;
                    }
                    double error = nn.getTestData().getErrorMargin(
                            nn.getInputNeurons(), result[i], i);
                    nn.getOutputNeurons()[i].setErr(error);
                }
                if (tempCorrect == nn.getTestData().getClassifications()) {
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

                //update learning rates for hidden nodes
                for (Weight[] r : nn.getItoH()) {
                    for (Weight w : r) {
                        //if sign has not changed
                        if (Math.signum(w.getGradient()) == 
                                Math.signum(w.getPrevGradient())) {
                            w.setAlpha(w.getAlpha() + K);
                        }else{ //if sign has changed
                            w.setAlpha(w.getAlpha() * (1-D));
                        }
                    }
                }

                //update learning rates for output nodes
                for (Weight[] r : nn.getHtoO()) {
                    for (Weight w : r) {
                        //if sign has not changed
                        if (Math.signum(w.getGradient()) == 
                                Math.signum(w.getPrevGradient())) {
                            w.setAlpha(w.getAlpha() + K);
                        }else{ //if sign has changed
                            w.setAlpha(w.getAlpha() * (1-D));
                        }
                    }
                }
            }
            sr = correct / (nn.getTestData().getXSize());
            System.out.println("Epoch: " + epoch + ", Success rate: " + sr);
            if (sr == 1.0) {
                break;
            }
            epoch++;

            nn.getTestData().scramble();
        }
        System.out.println("Trained with " + epoch * 
                nn.getTestData().getXSize() + " examples!");
    }

}

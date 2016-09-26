/*
 * .
 */
package hmm;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;

import org.apache.mahout.classifier.sequencelearning.hmm.HmmEvaluator;
import org.apache.mahout.classifier.sequencelearning.hmm.HmmModel;
import org.apache.mahout.classifier.sequencelearning.hmm.HmmTrainer;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
 

	
public class HMMTrainerTest {
	public static void main(String[] argsx) throws FileNotFoundException, IOException {
	
		System.out.println("start");
		HmmModel model;
		 
		// intialize the hidden/output state names
	    String[] hiddenNames = {"H0", "H1", "H2", "H3"};
	    String[] outputNames = {"O0", "O1", "O2"};
	    // initialize the transition matrix
	    double[][] transitionP = {{0.5, 0.1, 0.1, 0.3}, {0.4, 0.4, 0.1, 0.1},
	        {0.1, 0.0, 0.8, 0.1}, {0.1, 0.1, 0.1, 0.7}};
	    // initialize the emission matrix
	    double[][] emissionP = {{0.8, 0.1, 0.1}, {0.6, 0.1, 0.3},
	        {0.1, 0.8, 0.1}, {0.0, 0.1, 0.9}};
	    // initialize the initial probability vector
	    double[] initialP = {0.2, 0.1, 0.4, 0.3};
	    // now generate the model
	    model = new HmmModel(new DenseMatrix(transitionP), new DenseMatrix(
	        emissionP), new DenseVector(initialP));
	    model.registerHiddenStateNames(hiddenNames);
	    model.registerOutputStateNames(outputNames);
		
 
		 
    int[] observed = {1, 0, 2, 2, 0, 0, 1, 1, 1, 0, 2, 0, 1, 0, 1, 2, 1, 0, 1, 0, 1};

    
    HmmModel trained = HmmTrainer.trainBaumWelch(model, observed, 0.1, 10, false);

    Vector P = trained.getInitialProbabilities();
    Matrix B = trained.getEmissionMatrix();
    Matrix A = trained.getTransitionMatrix();

    System.out.println("P is:"); 
    System.out.println(P); 
    System.out.println("A is:"); 
    System.out.println(A); 
    System.out.println("B is:"); 
    System.out.println(B); 
    
    //预测后续4步的序列
    int[] decode1 = HmmEvaluator.predict(model, 4);
    System.out.println(Arrays.toString(decode1));
    
    //观测序列 0,1,2的概率
    double d2 = HmmEvaluator.modelLikelihood(model, new int[]{0,2,1}, false);
    System.out.println(d2);

    //给出观测数据0,1,2,1，计算其对应的最可能的隐藏状态   (使用了viterbi算法)
    int[] decode2 = HmmEvaluator.decode(model, new  int[]{0,1,2,1}, false);
    System.out.println(Arrays.toString(decode2));
    
	}
}

 
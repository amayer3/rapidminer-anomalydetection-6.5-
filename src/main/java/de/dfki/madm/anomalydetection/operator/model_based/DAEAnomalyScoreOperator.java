package de.dfki.madm.anomalydetection.operator.model_based;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.logging.Level;

import org.deeplearning4j.datasets.iterator.DoublesDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.primitives.Pair;

import com.rapidminer.example.Attributes;
import com.rapidminer.example.ExampleSet;
import com.rapidminer.operator.OperatorDescription;
import com.rapidminer.operator.OperatorException;
import com.rapidminer.operator.UserError;
import com.rapidminer.parameter.ParameterType;
import com.rapidminer.parameter.ParameterTypeBoolean;
import com.rapidminer.parameter.ParameterTypeCategory;
import com.rapidminer.parameter.ParameterTypeDouble;
import com.rapidminer.parameter.ParameterTypeInt;
import com.rapidminer.parameter.ParameterTypeList;
import com.rapidminer.parameter.ParameterTypeString;
import com.rapidminer.parameter.conditions.BooleanParameterCondition;
import com.rapidminer.parameter.conditions.EqualTypeCondition;
import com.rapidminer.tools.LogService;

import de.dfki.madm.anomalydetection.operator.AbstractAnomalyDetectionOperator;

/**
 * Operator description here ...
 * 
 * @author Alexander Mayer
 *
 */
public class DAEAnomalyScoreOperator extends AbstractAnomalyDetectionOperator {
	
	public final String PARAMETER_LEARNINGALGO = "learning algorithm";
	public static final String PARAMETER_HIDDEN_LAYERS = "hidden layers";
	public static final String PARAMETER_LEARNINGRATE = "learning rate";
	public static final String PARAMETER_ACTIVATION = "activation function";
	public static final String PARAMETER_EPOCHS = "epochs";
	public static final String PARAMETER_TRAINING_ITERATIONS = "training iterations";
	public static final String PARAMETER_WEIGHTINIT = "weight initialization";
	public static final String PARAMETER_UPDATER = "updater";
	public static final String PARAMETER_SETUP_AUTO = "use default autoencoder architecture"; 
	public static final String PARAMETER_REGULARIZATION_L2 = "l2 regularization";
	public static final String PARAMETER_REGULARIZATION_DROPOUT = "dropout regularization";
	public static final String PARAMETER_OUTER_LAYERS_SIGMOID = "set activation function of outer layers to SIGMOID"; 
	
	private static String[] activations = { "CUBE", "ELU", "HARDSIGMOID", "HARDTANH", "IDENTITY", "LEAKYRELU", "RATIONALTANH", "RELU", "RELU6",
			"RRELU", "SIGMOID", "SOFTMAX", "SOFTPLUS", "SOFTSIGN", "TANH", "RECTIFIEDTANH", "SELU", "SWISH",
			"THRESHOLDEDRELU", "GELU" };
	
	private static String[] weightInits = { "DISTRIBUTION", "ZERO", "ONES", "SIGMOID_UNIFORM", "NORMAL", "LECUN_NORMAL", "UNIFORM", "XAVIER", 
			"XAVIER_UNIFORM", "XAVIER_FAN_IN", "XAVIER_LEGACY", "RELU", "RELU_UNIFORM", "IDENTITY", "LECUN_UNIFORM", "VAR_SCALING_NORMAL_FAN_IN", 
			"VAR_SCALING_NORMAL_FAN_OUT", "VAR_SCALING_NORMAL_FAN_AVG", "VAR_SCALING_UNIFORM_FAN_IN", "VAR_SCALING_UNIFORM_FAN_OUT", 
			"VAR_SCALING_UNIFORM_FAN_AVG" };
		
	private static String[] optimizationAlgos = { "LINE_GRADIENT_DESCENT", "CONJUGATE_GRADIENT", "LBFGS", "STOCHASTIC_GRADIENT_DESCENT" };
	
	private static String[] updaters = { "Sgd",  "Nesterovs", "AdaGrad", "RmsProp", "Adam"};

	
	public DAEAnomalyScoreOperator(OperatorDescription description) {
		super(description);
	}
	
	// actual operator functionality
	@Override
	public double[] doWork(ExampleSet exampleSet, Attributes attributes,
			double[][] points) throws OperatorException {
		
		//check if input data is normalized
		double[] exampleValues = points[0];
		boolean normalized = true;
		
		for (int i = 0; i < exampleValues.length; i++) {	
			double value = exampleValues[i];
			if(value > 1 || value < 0) {
				normalized = false;	
				break;
			}
		}
		
		if(!normalized)
			LogService.getRoot().log(Level.WARNING, "Attribute values of the input data are not normalized to the value range [0, 1]. This may result in bad reconstruction results when the output layer outputs values between 0 and 1, e.g. when using the SIGMOID activation function.");
		
		int epochs = getParameterAsInt(PARAMETER_EPOCHS);
		int trainingIterations = getParameterAsInt(PARAMETER_TRAINING_ITERATIONS);
		
		// get dataset iterator
		int batchSize = exampleSet.size() / trainingIterations;
		DoublesDataSetIterator iterator = getIterator(points, batchSize);
		
		// get model configuration
		MultiLayerNetwork model = new MultiLayerNetwork(getModelConf(points[0].length));
		model.init(); 
        
        // train the model
        model.fit(iterator, epochs);
        
        INDArray inputSet = Nd4j.create(points);
        // Reconstruct all of the examples using the autoencoder
        INDArray outputSet = model.output(inputSet).castTo(DataType.DOUBLE);
        double[] outlierScore = new double[exampleSet.size()];
        
        for (int i=0; i<outlierScore.length; i++) {
            
        	INDArray inputData = inputSet.getRow(i);
        	INDArray outputData = outputSet.getRow(i);
            double euclideanDistance = inputData.distance2(outputData); //Transforms.euclideanDistance(d1, d2)?
            outlierScore[i] = euclideanDistance;
		}
        
		return outlierScore;
	}
	
	// create dataset iterator
	private static DoublesDataSetIterator getIterator(double[][] points, int batchSize) {
		
		ArrayList<Pair<double[], double[]>> featureLabelList = new ArrayList<>();
		
		// creating an iterable list of pairs (features, labels) for model training
		for(int i=0; i<points.length; i++) {
			
			double[] example = points[i];
			featureLabelList.add(new Pair<>(example, example));		
	    } 
		
		return new DoublesDataSetIterator(featureLabelList, batchSize);
	}
	
	// model configuration
	private MultiLayerConfiguration getModelConf(int numberOfAttributes) throws OperatorException {
	
		Activation activation = Activation.fromString(activations[getParameterAsInt(PARAMETER_ACTIVATION)]);
		OptimizationAlgorithm learningAlgorithm = OptimizationAlgorithm.valueOf(optimizationAlgos[getParameterAsInt(PARAMETER_LEARNINGALGO)]);
		WeightInit weightInit = WeightInit.valueOf(weightInits[getParameterAsInt(PARAMETER_WEIGHTINIT)]);
		double learningRate = getParameterAsDouble(PARAMETER_LEARNINGRATE);
		
		String updaterParam = updaters[getParameterAsInt(PARAMETER_UPDATER)];
		IUpdater updater = null;
		
		switch(updaterParam){
        case "Sgd":
            updater = new Sgd(learningRate);
        case "Nesterovs":
        	updater = new Nesterovs(learningRate, 0.9);
        case "AdaGrad":
        	updater = new AdaGrad(1e-1, 1e-6);
            break;
        case "RmsProp":
        	updater = new RmsProp(1e-1, 0.95, 1e-8);
            break;
        case "Adam":
        	updater = new Adam(1e-3, 0.9, 0.999, 1e-8);
            break;
        }
		
		boolean setup_auto = getParameterAsBoolean(PARAMETER_SETUP_AUTO);
		
		List<String[]> hiddenLayers = new ArrayList<String[]>();
		if(setup_auto) { //default is 3 hidden layers with number of neurons based on input data dimension
			int size_hiddenLayer1 = (int)Math.round(numberOfAttributes * 0.5); 
			int size_latentSpace = (int)Math.round(numberOfAttributes * 0.2);
			
			if(size_hiddenLayer1 == 0) 
				size_hiddenLayer1 = 1;
			if(size_latentSpace == 0) 
				size_latentSpace = 1;
			
			hiddenLayers.add(new String[] {"hidden layer 1", size_hiddenLayer1 + ""});
			hiddenLayers.add(new String[] {"latent space", size_latentSpace + ""});
		} else {
			hiddenLayers = getParameterList(PARAMETER_HIDDEN_LAYERS);
		}
		
		boolean outerLayers_sigmoid = getParameterAsBoolean(PARAMETER_OUTER_LAYERS_SIGMOID);
		
		Activation outerActivation;
		if(outerLayers_sigmoid)
			outerActivation = Activation.SIGMOID;
		else
			outerActivation = activation;
		
		boolean l2Reg = getParameterAsBoolean(PARAMETER_REGULARIZATION_L2);
		boolean dropoutReg = getParameterAsBoolean(PARAMETER_REGULARIZATION_DROPOUT);

		double l2RegVal = 0.0;
		double dropoutRegVal = 1.0;
		if(l2Reg)
			l2RegVal = 0.0001;
		if(dropoutReg)
			dropoutRegVal = 0.7;
		
		Layer[] layers = new Layer[hiddenLayers.size() * 2];
		int[] layerSizes = new int[hiddenLayers.size()]; //number of neurons from first hidden layer up to latent space
		
		int index = 0;
		int lastLayerSize = numberOfAttributes;
		Iterator<String[]> i = hiddenLayers.iterator();
		
		//--- Encoder including latent space ---
		while (i.hasNext()) {
			
			String[] nameSizePair = i.next();
			int layerSize = Integer.valueOf(nameSizePair[1]);
			layerSizes[index] = layerSize;
			
			//activation of the first hidden layer depends on parameter setting
			if(index == 0) { 
				
				layers[index] = new DenseLayer.Builder()
	    	    		.nIn(lastLayerSize)
	    	    		.nOut(layerSize)
	    	    		.activation(outerActivation) 
	    	            .build();
			}
			else {
				
				layers[index] = new DenseLayer.Builder()
	    	    		.nIn(lastLayerSize)
	    	    		.nOut(layerSize)
	    	    		.dropOut(dropoutRegVal)
	    	    		.activation(activation) 
	    	            .build();
			}
			
			lastLayerSize = layerSize;
			index++;
		}
		
		if(index < 2)
        	throw new UserError(this, 1002);
		
		//--- Decoder --- (mirroring encoding layers except latent space)
		for (int j = index - 2; j >= 0; j--) {
			
			int layerSize = layerSizes[j];
			
			layers[index] = new DenseLayer.Builder()
    	    		.nIn(lastLayerSize)
    	    		.nOut(layerSize)
    	    		.dropOut(dropoutRegVal)
    	    		.activation(activation) 
    	            .build();
			
			lastLayerSize = layerSize;
			index++;
		}
		
		//--- Output Layer --- (activation depends on parameter setting)
		layers[index] = new OutputLayer.Builder()
	    		.nIn(lastLayerSize)
	            .nOut(numberOfAttributes)
	            .activation(outerActivation) 
	    		.lossFunction(LossFunction.MSE)
	            .build();
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        	    .seed(12345)
        	    .optimizationAlgo(learningAlgorithm)
        	    .updater(updater)
    	        .weightInit(weightInit)
    	        .l2(l2RegVal)
        	    .list(layers)
        	    .backpropType(BackpropType.Standard)
        	    .build();
		
		if(setup_auto) { //log info message
			int size_hiddenLayer1 = layerSizes[0];
			int size_latentSpace = layerSizes[1];
			
			String info_message = "Autoencoder Architecture was set up automatically: size_hiddenLayer1 = " + size_hiddenLayer1 + ", size_latentSpace = " + size_latentSpace + ", size_hiddenLayer3 = " + size_hiddenLayer1;
			LogService.getRoot().log(Level.INFO, info_message);
		}
        
        return conf;
	}
	
	// specify parameters
	@Override
	public List<ParameterType> getParameterTypes() {
		
		List<ParameterType> types = super.getParameterTypes();
		
		types.add(new ParameterTypeBoolean(PARAMETER_SETUP_AUTO, "If this option is checked, the autoencoder model is set up automatically, i.e. it consists of 3 hidden layers and the number of neurons per hidden layer is dynamically set based on the input data dimension: size_hiddenLayer1 = dimension * 0.5, size_latentSpace = dimension * 0.2, size_hiddenLayer3 = size_hiddenLayer1. The size of the hidden layers is logged as info message to the console.", true, false));

		ParameterType type = new ParameterTypeList(
				PARAMETER_HIDDEN_LAYERS,
				"Defines the name and size of all encoding hidden layers (from first hidden layer up to latent space). The decoding part will be automatically set up by mirroring the encoding hidden layers. Please note that a Deep Autoencoder must consist of at least 3 hidden layers, meaning that at least 2 list entries are mandatory.",
				new ParameterTypeString("hidden_layer_name", "The name of the hidden layer.", "hidden layer", false),
				new ParameterTypeInt(
						"hidden_layer_size",
						"The size of the hidden layer.",
						1, Integer.MAX_VALUE, 1, false),
				new ArrayList<String[]>(), false);
		type.registerDependencyCondition(new BooleanParameterCondition(this, PARAMETER_SETUP_AUTO, true, false));
		types.add(type);
						
		types.add(new ParameterTypeCategory(PARAMETER_ACTIVATION, "The activation function is responsible for transforming the summed weighted input of a neuron into the activation or output of that neuron. The here defined activation function is referred to all layers' neurons (except input layer).", activations, 10, false));
		
		types.add(new ParameterTypeBoolean(PARAMETER_OUTER_LAYERS_SIGMOID, "If this option is checked, the activation function is set to SIGMOID at the two ends of the network (first hidden layer and output layer).", true, false));

		types.add(new ParameterTypeCategory(PARAMETER_LEARNINGALGO, "The learning algorithm specifies how weight updates are made during training.", optimizationAlgos, 3, true));
		
		types.add(new ParameterTypeCategory(PARAMETER_UPDATER, "Updaters differ in how they help optimizing the learning rate until the neural network converges on its most performant state.", updaters, 0, false));
		
		ParameterTypeDouble doubleType = (new ParameterTypeDouble(PARAMETER_LEARNINGRATE, "The value that defines to which extent the weights are updated during training.", 0.000001, 0.1, 0.001, false));
		doubleType.registerDependencyCondition(new EqualTypeCondition(this, PARAMETER_UPDATER, updaters, true, 0, 1));
		types.add(doubleType);
		
		types.add(new ParameterTypeInt(PARAMETER_EPOCHS, "The value that sets the number of epochs to train.", 1, Integer.MAX_VALUE, 100, false));
		
		types.add(new ParameterTypeInt(PARAMETER_TRAINING_ITERATIONS, "The value that indicates how often weights are updated during one epoch of training.", 1, Integer.MAX_VALUE, 1, true));
			
		types.add(new ParameterTypeCategory(PARAMETER_WEIGHTINIT, "", weightInits, 7, true));
		
		types.add(new ParameterTypeBoolean(PARAMETER_REGULARIZATION_L2, "", false, true));
		
		types.add(new ParameterTypeBoolean(PARAMETER_REGULARIZATION_DROPOUT, "", false, true));
		
		return types;
	}
}

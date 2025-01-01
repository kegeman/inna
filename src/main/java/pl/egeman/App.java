package pl.egeman;

public class App 
{
    public static void main( String[] args ) {
        System.out.println("The Inna Trainer is running...");
        System.out.println("The current directory is: " + System.getProperty("user.dir"));
        MainParameters params = new MainParameters();
        params.samplesDirPathName = "/home/charlie/projects/neural/data/samples"; // flat neural network input files
        params.labelsDirPathName = "/home/charlie/projects/neural/data/labels"; // flat neural network labels files
        params.modelFileName = "/home/charlie/projects/neural/data/model";
        params.minSamplesCount = 500;
        // TODO: params.initialize(args);
        Main.create(params).run();
        System.out.println("The Inna Trainer has stopped");
    }
}

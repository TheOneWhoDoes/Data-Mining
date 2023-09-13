/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author AVAJANG-PC
 */
import weka.associations.Apriori;
import weka.associations.FPGrowth;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;



public class AssociationRules {
   public static void main(String args[])throws Exception{
    System.out.println("Weka loaded.");
    //load dataset
    String dataset = "D:/Weka-3-8/data/supermarket.arff";
    DataSource source = new DataSource(dataset);
    //get instance object
    Instances data=source.getDataSet();
    
    //apriori algorithm
    Apriori apriori_model=new Apriori();
    FPGrowth fpgrowth_model=new FPGrowth();
    //build model
    apriori_model.buildAssociations(data);
    fpgrowth_model.buildAssociations(data);
    System.out.println("apriori_model:");
    System.out.println(apriori_model);
    System.out.println("***********************************"); 
    System.out.println();   
    System.out.println("fpgrowth_model:");
    System.out.println(fpgrowth_model);   
    System.out.println("");
    
   }
   
  } 


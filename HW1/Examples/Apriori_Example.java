/*
 * First you have to add weka.jar to your project external libraries
 */

import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;

import weka.associations.AssociationRule;
import weka.associations.AssociationRules;
import weka.associations.NominalItem;
import weka.associations.Apriori;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Apriori_Example {
    public static void main(String[] args)throws Exception{
        System.out.println("Weka loaded.");
        //load dataset

        // String dataset = "C:\\Program Files\\Weka-3-8\\data\\supermarket.arff";
        String dataset = "/home/mmh/weka-3-8-3/data/supermarket.arff";

        DataSource source = new DataSource(dataset);
        //get instance object
        Instances data=source.getDataSet();

        Apriori apriori_model = new Apriori();
        String[] options = { "-N", "10000", "-C", "0.9", "-M", "0.1", "-S", "0.8"};
        apriori_model.setOptions(options);
        apriori_model.buildAssociations(data);
        AssociationRules ARS = apriori_model.getAssociationRules();
        List<AssociationRule> ruleList = ARS.getRules();

        for(int i = 0; i < ruleList.size(); i++) {

            AssociationRule AR = ruleList.get(i);

            Collection premise = AR.getPremise();
            int premiseSupport = AR.getPremiseSupport();

            Collection consequence = AR.getConsequence();
            int consequenceSupport = AR.getConsequenceSupport();

            int totalSupport = AR.getTotalSupport();
            Collection<NominalItem> frequentPattern = new HashSet<>();

            Iterator iterator = premise.iterator();
            while(iterator.hasNext()) {
                frequentPattern.add((NominalItem)iterator.next());
            }

            iterator = consequence.iterator();
            while(iterator.hasNext()) {
                frequentPattern.add((NominalItem)iterator.next());
            }

            System.out.println("*****************************************");
            System.out.println(premise + " : " + premiseSupport);
            System.out.println(consequence + " : " + consequenceSupport);
            System.out.println(frequentPattern + " : " + totalSupport);
            System.out.println("*****************************************");


        }

    }
}

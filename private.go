package DecisionTree

import (
	"math"
	"reflect"

	"github.com/SamuelCarroll/DataTypes"
)

// var classes []ClassAvg
// var classSamples [][]*dataTypes.Data

//avgClass will average all attributes for all classes and return the running average
//and standard deviation
func avgClass(allData []*dataTypes.Data, classSamples [][]*dataTypes.Data, classes []ClassAvg) {
	//for each piece of data adjust the class by one and find the running average for that
	//classes attributes
	for _, datum := range allData {
		classIndex := datum.Class - 1

		//set the averages to the running average, increase class count and append the datum
		//to the appropriate class sample array
		classes[classIndex].averages = runningAvg(classes[classIndex].averages, *datum, classes[classIndex].count)
		classes[classIndex].count++
		classSamples[classIndex] = append(classSamples[classIndex], datum)
	}

	//Find the standard deviation after all averages are found
	for i, class := range classes {
		classes[i].stdDev = findStds(classSamples[i], class)
	}
}

//getMajority will find the majority class for whatever data is passed into it (limited
// by the number of classes we have)
func getMajority(data []*dataTypes.Data, numClasses int) int {
	counts := make([]int, numClasses)

	//count each occurance of a class
	for _, datum := range data {
		counts[datum.Class-1]++
	}

	//set max class index to zero and compare all the class count
	max := 0
	for i := 1; i < numClasses; i++ {
		if counts[i] > counts[max] {
			max = i
		}
	}

	//return class majority (remember to add one for off by one error)
	return max + 1
}

//stoppingCond will check if we have reached the desired purity of a set
func stoppingCond(nodeData []*dataTypes.Data, stopCond float64, classes int) bool {
	count := make([]int, classes)
	percent := make([]float64, classes)

	//Count each occurance of a class so we can check the purity
	for _, elem := range nodeData {
		count[elem.Class-1]++
	}

	//for each class count check the percentage of each class count to see if we have
	//reached the threshold
	for i := range count {
		percent[i] = float64(count[i]) / float64(len(nodeData))
		if percent[i] >= stopCond {
			return true
		}
	}

	return false
}

//findEntropy will check the entropy given an average and standard deviation as a split
// value and
func findEntropy(valueIndex, classCount int, avg, stdDev float64, nodeData []*dataTypes.Data) float64 {
	var classInstances []float64
	var classEntropies []float64
	var classWeights []float64

	//initialze class count, entropy and weights to zero
	for i := 0; i < classCount; i++ {
		classInstances = append(classInstances, 0.0)
		classEntropies = append(classEntropies, 0.0)
		classWeights = append(classWeights, 0.0)
	}

	//for each datum in our dataset we should get the attribute we are splitting on, and
	//that datums class, add a 1 or 0 based on how the split value affects that datum
	for _, datum := range nodeData {
		instance := getVal(datum.FeatureSlice[valueIndex])
		classIndex := datum.Class - 1

		classInstances[classIndex] += countClass(instance, avg+stdDev)
	}

	//find the length of all data to set the class weights
	lenData := float64(len(nodeData))

	entropy := 0.0
	for i := 0; i < classCount; i++ {
		//Assuming we have at least one class instance we can find purity of that split
		if classInstances[i] > 0 {
			classWeights[i] = classInstances[i] / lenData
			classEntropies[i] = classWeights[i] * math.Log2(classWeights[i])
			entropy += classWeights[i] * classEntropies[i]
		}
	}

	return entropy * -1
}

//countClass will check which direction the split value will affect a value,
//returns 1 if we go left, and 0 if we go right
func countClass(instance float64, splitVal float64) float64 {
	if instance <= splitVal {
		return 1
	}

	return 0
}

//initializeAvgs will initialize an average value given the type of an attribute for all
//attributes we have
func initializeAvgs(example dataTypes.Data) []interface{} {
	var newAvgVals []interface{}

	//switch on the variable type of an attribute
	for i := range example.FeatureSlice {
		switch example.FeatureSlice[i].(type) {
		//for both floats and bools set initial value to 0 (false if it's bool)
		case float64:
			newAvgVals = append(newAvgVals, 0.0)
		case bool:
			newAvgVals = append(newAvgVals, 0.0)
		//for a string initialize the initial value to an empty string, I still need to
		//figure out a good way to do this
		case string:
			newAvgVals = append(newAvgVals, "")
		}
	}

	return newAvgVals
}

//findLeast will find the smallest value in a floating point value array
func findLeast(values []float64) (int, float64) {
	leastIndex := 0
	leastVal := values[0]

	//for each value in the value array check if it's less than the current minimum,
	//if it is reset current minimum and current minimum index
	for i, val := range values {
		if val < leastVal {
			leastVal = val
			leastIndex = i
		}
	}

	return leastIndex, leastVal
}

//runningAvg will calculate the running average of a generic interface, given a new
//piece of datum, and count of pervious data points
func runningAvg(oldAvgs []interface{}, newVal dataTypes.Data, n int) []interface{} {
	//if we have an empty inital average we need to initialize the average values
	if len(oldAvgs) < len(newVal.FeatureSlice) {
		oldAvgs = initializeAvgs(newVal)
	}

	//for every attribute calculate an approximate running sum value, add the new value
	//then divide the approximate running sum value by one greater than count of
	//contributing data points
	for i := range newVal.FeatureSlice {
		//keep track of the running averages
		temp := getVal(oldAvgs[i]) * float64(n)
		temp += getVal(newVal.FeatureSlice[i])
		oldAvgs[i] = temp / float64(n+1)
	}

	//return the new running average
	return oldAvgs
}

//findStds will find the standard deviation given a set of data points and a list
//of attribute averages for those data points
func findStds(classSam []*dataTypes.Data, class ClassAvg) []interface{} {
	var stdDev []interface{}

	//if we don't have any instances just return an empty interface
	if len(classSam) == 0 {
		return stdDev
	}

	//get count of attributes
	featureLen := len(classSam[0].FeatureSlice)

	//for every attribute in a list of data points find the standard deviation in the usual way
	for i := 0; i < featureLen; i++ {
		classTotal := 0.0
		for _, sample := range classSam {
			class.stdDev = append(class.stdDev, 0.0)
			//reflect the type of the feature slice index handle float, bool and string (don't worry about bool and str yet)
			sampleVal := getVal(sample.FeatureSlice[i])
			classVal := getVal(class.averages[i])
			classTotal += math.Pow((sampleVal - classVal), 2)
		}

		//add the standard deviation to the standard deviation list
		stdDev = append(stdDev, classTotal/float64(class.count))
	}

	return stdDev
}

//findIndex will return the attribute index to be used for our split value
func findIndex(entropyVals []float64) int {
	minVal := entropyVals[0]
	minIndex := 0

	//for every split value (same as the entropy value) find the smallest entropy, highest
	// purity of split values
	for i, contender := range entropyVals {
		if contender < minVal {
			minIndex = i
			minVal = contender
		}
	}

	return minIndex
}

//GetFloatReflectVal takes an interface value and returns it as a float64 type
func GetFloatReflectVal(val interface{}) float64 {
	v := reflect.ValueOf(val)
	v = reflect.Indirect(v)

	floatVal := v.Convert(reflect.TypeOf(0.0))
	return floatVal.Float()
}

//GetBoolReflectVal takes an interface value and returns it as a bool type
func GetBoolReflectVal(val interface{}) float64 {
	v := reflect.ValueOf(val)
	v = reflect.Indirect(v)

	boolVal := v.Convert(reflect.TypeOf(true))

	if boolVal.Bool() == true {
		return 1.0
	}
	return 0.0
}

//GetStrReflectVal takes an interface value and returns it as a string value
func GetStrReflectVal(val interface{}) string {
	v := reflect.ValueOf(val)
	v = reflect.Indirect(v)

	strVal := v.Convert(reflect.TypeOf(""))
	return strVal.String()
}

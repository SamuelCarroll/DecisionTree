package DecisionTree

import (
	"math"
	"reflect"

	"github.com/SamuelCarroll/DataTypes"
)

// var classes []ClassAvg
// var classSamples [][]*dataTypes.Data

func avgClass(allData []*dataTypes.Data, classSamples [][]*dataTypes.Data, classes []ClassAvg) {
	for _, datum := range allData {
		classIndex := datum.Class - 1

		classes[classIndex].averages = runningAvg(classes[classIndex].averages, *datum, classes[classIndex].count)
		classes[classIndex].count++
		classSamples[classIndex] = append(classSamples[classIndex], datum)
	}

	for i, class := range classes {
		classes[i].stdDev = findStds(classSamples[i], class)
	}
}

func getMajority(data []*dataTypes.Data) int {
	count1, count2, count3 := 0, 0, 0
	for _, datum := range data {
		if datum.Class == 1 {
			count1++
		} else if datum.Class == 2 {
			count2++
		} else {
			count3++
		}
	}
	if count1 > count2 && count1 > count3 {
		return 1
	} else if count2 > count1 && count2 > count3 {
		return 2
	} else {
		return 3
	}
}

func stoppingCond(nodeData []*dataTypes.Data, stopCond float64) bool {
	var count [3]int
	var percent [3]float64

	for _, elem := range nodeData {
		count[elem.Class-1]++
	}

	for i := range count {
		percent[i] = float64(count[i]) / float64(len(nodeData))
		if percent[i] >= stopCond {
			return true
		}
	}

	return false
}

func findEntropy(valueIndex, classCount int, avg, stdDev float64, nodeData []*dataTypes.Data) float64 {
	var classInstances []float64
	var classEntropies []float64
	var classWeights []float64

	for i := 0; i < classCount; i++ {
		classInstances = append(classInstances, 0.0)
		classEntropies = append(classEntropies, 0.0)
		classWeights = append(classWeights, 0.0)
	}

	for _, datum := range nodeData {
		instance := GetFloatReflectVal(datum.FeatureSlice[valueIndex])
		classIndex := datum.Class - 1

		classInstances[classIndex] += countClass(instance, avg+stdDev)
	}

	lenData := float64(len(nodeData))

	entropy := 0.0
	for i := 0; i < classCount; i++ {
		if classInstances[i] > 0 {
			classWeights[i] = classInstances[i] / lenData
			classEntropies[i] = classWeights[i] * math.Log2(classWeights[i])
			entropy += classWeights[i] * classEntropies[i]
		}
	}

	return entropy * -1
}

func countClass(instance float64, splitVal float64) float64 {
	if instance < splitVal {
		return 1
	}

	return 0
}

func initializeAvgs(example dataTypes.Data) []interface{} {
	var newAvgVals []interface{}

	for i := range example.FeatureSlice {
		switch example.FeatureSlice[i].(type) {
		case float64:
			newAvgVals = append(newAvgVals, 0.0)
		case bool:
			newAvgVals = append(newAvgVals, false)
		case string:
			newAvgVals = append(newAvgVals, "")
		}
	}

	return newAvgVals
}

//TODO generalize this
func findLeast(values []float64) (int, float64) {
	leastIndex := 0
	leastVal := values[0]

	for i, val := range values {
		if val < leastVal {
			leastVal = val
			leastIndex = i
		}
	}

	return leastIndex, leastVal
}

func runningAvg(oldAvgs []interface{}, newVal dataTypes.Data, n int) []interface{} {
	if len(oldAvgs) < len(newVal.FeatureSlice) {
		oldAvgs = initializeAvgs(newVal)
	}

	for i := range newVal.FeatureSlice {
		//reflect the type of the feature slice index handle float, bool and string (don't worry about bool and str yet)
		switch val := oldAvgs[i].(type) {
		case float64:
			temp := float64(val) * float64(n)
			temp += GetFloatReflectVal(newVal.FeatureSlice[i])
			oldAvgs[i] = temp / float64(n+1)
		}
	}

	return oldAvgs
}

func findStds(classSam []*dataTypes.Data, class ClassAvg) []interface{} {
	var stdDev []interface{}

	if len(classSam) == 0 {
		return stdDev
	}

	featureLen := len(classSam[0].FeatureSlice)

	for i := 0; i < featureLen; i++ {
		classTotal := 0.0
		for _, sample := range classSam {
			class.stdDev = append(class.stdDev, 0.0)
			//reflect the type of the feature slice index handle float, bool and string (don't worry about bool and str yet)
			fsample := GetFloatReflectVal(sample.FeatureSlice[i])
			fclass := GetFloatReflectVal(class.averages[i])
			classTotal += math.Pow((fsample - fclass), 2)
		}

		stdDev = append(stdDev, classTotal/float64(class.count))
	}

	return stdDev
}

func findIndex(entropyVals []float64) int {
	minVal := entropyVals[0]
	minIndex := 0

	for i, contender := range entropyVals {
		if contender < minVal {
			minIndex = i
			minVal = contender
		}
	}

	return minIndex
}

func GetFloatReflectVal(val interface{}) float64 {
	v := reflect.ValueOf(val)
	v = reflect.Indirect(v)

	floatVal := v.Convert(reflect.TypeOf(0.0))
	return floatVal.Float()
}

func GetBoolReflectVal(val interface{}) bool {
	v := reflect.ValueOf(val)
	v = reflect.Indirect(v)

	boolVal := v.Convert(reflect.TypeOf(true))
	return boolVal.Bool()
}

func GetStrReflectVal(val interface{}) string {
	v := reflect.ValueOf(val)
	v = reflect.Indirect(v)

	strVal := v.Convert(reflect.TypeOf(""))
	return strVal.String()
}

//////////////////////////////////////////////////////////////////////////////
// Example illustrating the use of GCoptimization.cpp
//
/////////////////////////////////////////////////////////////////////////////
//
//  Optimization problem:
//  is a set of sites (pixels) of width 10 and hight 5. Thus number of pixels is 50
//  grid neighborhood: each pixel has its left, right, up, and bottom pixels as neighbors
//  7 labels
//  Data costs: D(pixel,label) = 0 if pixel < 25 and label = 0
//            : D(pixel,label) = 10 if pixel < 25 and label is not  0
//            : D(pixel,label) = 0 if pixel >= 25 and label = 5
//            : D(pixel,label) = 10 if pixel >= 25 and label is not  5
// Smoothness costs: V(p1,p2,l1,l2) = min( (l1-l2)*(l1-l2) , 4 )
// Below in the main program, we illustrate different ways of setting data and smoothness costs
// that our interface allow and solve this optimizaiton problem

// For most of the examples, we use no spatially varying pixel dependent terms. 
// For some examples, to demonstrate spatially varying terms we use
// V(p1,p2,l1,l2) = w_{p1,p2}*[min((l1-l2)*(l1-l2),4)], with 
// w_{p1,p2} = p1+p2 if |p1-p2| == 1 and w_{p1,p2} = p1*p2 if |p1-p2| is not 1

#include <ctype.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <string>
#include <time.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/filesystem.hpp>
#include <limits>

//#include <opencv/contrib/contrib.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;
using namespace ximgproc;
//using namespace GCoptimization;

#include "GCoptimization.h"
#include "IcgBench.h"
#include "CImg.h"

using namespace cimg_library;

struct ForDataFn {
	int numLab;
	int *data;
};

struct Result {
	int *labels;
	long long e;
};

int delta = 5;
int desvio = 1.2;

int smoothFn(int p1, int p2, int l1, int l2) {

	//return exp( pow(p1 - p2, 2)/(2*d*d) );

	if (l1 == l2) {
		return 0;
	} else {
		//return delta * abs(p1 - p2);//pow(p1 - p2, 2);
		return delta * exp(-pow(p1 - p2, 2) / (2 * desvio * desvio));
	}

	//return a * abs(p1 - p2);
	//if ((l1 - l2) * (l1 - l2) <= 4)
	// return ((l1 - l2) * (l1 - l2));
	// else
	// return (4);
}

int dataFn(int p, int l, void *data) {
	ForDataFn *myData = (ForDataFn *) data;
	int numLab = myData->numLab;

	return (myData->data[p * numLab + l]);
}

int loadImage(std::string path, Mat& mask, Mat &labelSuperpixels) {

	//Load Image with opencv
	Mat image;

	image = imread(path, CV_LOAD_IMAGE_COLOR);   // Read the file

	if (!image.data)                              // Check for invalid input
	{
		printf("Could not open or find the image\n");
		return 0;
	}

	//cvtColor(image, gray, CV_RGB2GRAY);

	int num_iterations = 4;
	int prior = 2;
	bool double_step = false;
	int num_superpixels = 100;
	int num_levels = 4;
	int num_histogram_bins = 5;

	Mat result;
	Ptr<SuperpixelSEEDS> seeds;
	int width, height;

	width = image.size().width;
	height = image.size().height;

	seeds = createSuperpixelSEEDS(width, height, image.channels(),
			num_superpixels, num_levels, prior, num_histogram_bins,
			double_step);

	Mat converted;
	cvtColor(image, converted, COLOR_BGR2HSV);

	double t = (double) getTickCount();


	Ptr<SuperpixelSLIC> slic = createSuperpixelSLIC(converted,SLIC,50,float(100));


	//seeds->iterate(converted, num_iterations);
	slic->iterate(num_iterations);
	result = image;

	t = ((double) getTickCount() - t) / getTickFrequency();
	//printf("SEEDS segmentation took %i ms with %3i superpixels\n",
	//		(int) (t * 1000), seeds->getNumberOfSuperpixels());

	/* retrieve the segmentation result */
	//Mat labels;
	//seeds->getLabels(labels);
	//slic->getLabels(labels);
	/* get the contours for displaying */
	slic->getLabelContourMask(mask, false);
	result.setTo(Scalar(50, 50, 255), mask);


	//seeds->getLabels()


	slic->getLabels(labelSuperpixels);



	//cout << labelSuperpixels.size() << endl;

	//abort();
	//imshow("Superpixel", result);

	//namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	//imshow("Display window", superPixel);           // Show our image inside it.
	//imshow("Gray image", gray);

	//waitKey(0);

	// Wait for a keystroke in the window

	return slic->getNumberOfSuperpixels();

}

////////////////////////////////////////////////////////////////////////////////
// in this version, set data and smoothness terms using arrays
// grid neighborhood structure is assumed
//
Result * GridGraph_DMAT(const Mat& I, int num_labels, int f_labels[],
		std::vector<IcgBench::Seed> seeds, int countSeed[], CImg<float> *dataTermCuda,
		vector<vector<IcgBench::Seed>> &super) {

	int width = I.cols;
	int height = I.rows;
	int num_pixels = width * height;
	long long e = 0;

	int *result = new int[num_pixels];   // stores result of optimization

	vector<Mat> labelsSeed(num_labels);
	int countInstSeed[num_labels];

	for (int i = 0; i < num_labels; ++i) {
		Mat l(3, countSeed[i], CV_8SC1);
		labelsSeed[i] = l;
		countInstSeed[i] = 0;
	}

	for (unsigned int i = 0; i < seeds.size(); ++i) {
		IcgBench::Seed seed = seeds[i];
		unsigned int p = I.at<unsigned char>(seed.y, seed.x);
		//printf("%u", p);
		(labelsSeed[seed.label]).at<unsigned char>(0, countInstSeed[seed.label]) =
				p;
		(labelsSeed[seed.label]).at<unsigned char>(1, countInstSeed[seed.label]) =
				seed.y;
		(labelsSeed[seed.label]).at<unsigned char>(2, countInstSeed[seed.label]) =
				seed.x;

		countInstSeed[seed.label]++;
	}

	Mat r = I.reshape(1, num_pixels);
	//cout << r.depth() << ", " << r.channels() << endl;
	//printf("%d", r.);
	// first set up the array for data costs
	int sizeData = num_pixels * num_labels;
	double *data = new double[sizeData];
	int *dataInt = new int[sizeData];
	double min = std::numeric_limits<double>::max();
	double max = std::numeric_limits<double>::min();
	for (int i = 0; i < num_pixels; i++) {
		for (int l = 0; l < num_labels; l++) {
			//printf("Teste %d\n", r.at<unsigned char>(i));
			//data[i * num_labels + l] = pow(
			//		f_labels[l] - r.at<unsigned char>(0, i), 2);
			dataInt[i * num_labels + l] = abs(
					f_labels[l] - r.at<unsigned char>(0, i));

			/*int x = i/I.size().width;
			 int y = i%I.size().width;
			 Mat t1(1,countSeed[l],CV_32F);
			 Mat t2(1,countSeed[l],CV_32F);
			 Mat rootSquare;
			 pow(x - labelsSeed[l].col(2), 2, t1);
			 pow(y - labelsSeed[l].col(1), 2, t2);
			 sqrt(t1 + t2, rootSquare);*/

			//cout << labelsSeed[l].col(0) << endl;
			//abort();
			Mat c = labelsSeed[l].col(0);
			int diff = 0;
			for (int j = 0; j < c.size().height; j++) {
				diff += abs(r.at<unsigned char>(0, i) - c.at<unsigned char>(j, 0));
				//printf("%d", diff);
			}
			double p = diff / (1.0 * countSeed[l]);

			data[i * num_labels + l] = p;

			if (p < min) {
				min = p;
				//printf("%lf\n", min);
			}
			if (p > max) {
				max = p;
				//printf("%lf\n", min);
			}

			/*	double p = sum((r.at<unsigned char>(0, i) - labelsSeed[l].col(0)))[0]/countSeed[l];
			 //printf("%lf\n", -log());
			 data[i * num_labels + l] = p;
			 if ( p < min){
			 min = p;
			 //printf("%lf\n", min);
			 }
			 if ( p > max){
			 max = p;
			 //printf("%lf\n", min);
			 }*/

		}
	}

	//min = abs(min);
	double sumData = 0;
	for (int i = 0; i < sizeData; ++i) {
		data[i] = (data[i] - min)/(max - min);
		//printf("%lf\n", data[i]);
		//sumData += data[i];
	}

	//int *dataInt = new int[sizeData];

	for (int i = 0; i < sizeData; ++i) {
		//printf("%lf\n", -log(data[i]));
		data[i] = abs(-1* -log(data[i]));
		//dataInt[i] = data[i];
		//printf("%d\n", dataInt[i]);
	}

	for (int i = 0; i < I.size().height; i++) {
			for (int j = 0; j < I.size().width; j++) {
				for (int l = 0; l < num_labels; l++) {

					dataInt[(i * I.size().width + j) * num_labels + l] = (*dataTermCuda)(j,i, 0, l);
				}
			}
		}

	/*for (int i = 0; i < num_pixels; i++) {
		for (int l = 0; l < num_labels; l++) {

			dataInt[i * num_labels + l] = (*dataTermCuda)(i/num_pixels, i%num_pixels, 0, l);//(*dataTermCuda)();
		}
	}*/

	try {
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width, height,
				num_labels);

// set up the needed data to pass to function for the data costs
		ForDataFn toFn;
		toFn.data = dataInt;
		toFn.numLab = num_labels;

		gc->setDataCost(&dataFn, &toFn);

// smoothness comes from function pointer
		gc->setSmoothCost(&smoothFn);

//gc->setLabelCost()
		//gc->setLabelOrder(true);

		//gc->setLabelOrder(true);
		//gc->dynamic_programming();

		int *ordem = new int(num_labels);
		for (int i = 0; i < num_labels; ++i) {
			ordem[i] = num_labels - i - 1;
		}
		gc->setLabelOrder(ordem, num_labels);

//printf("\nBefore optimization energy is %lld", gc->compute_energy());
//for(int i =0; i < 100; i+=10){
		//gc->expansion(1); // run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		srand(time(NULL));
//gc->swap(10);

		gc->dynamic_programming(super, super.size(), dataTermCuda);
		//gc->expansion(1);

		//printf("\n%lld", gc->compute_energy());

		for (int i = 0; i < num_labels; ++i) {
			ordem[i] = i;
		}
		gc->setLabelOrder(ordem, num_labels);


		//gc->expansion(1);

		//gc->setLabel(0,0);

		//printf("\n%lld", gc->compute_energy());

//printf(" Dynamic Programming ");
//gc->dynamic_programming();
//printf("\n%lld", gc->compute_energy());
//}
//gc->expansion(100); // run expansion for 2 iterations. For swap use gc->swap(num_iterations);
//printf("\nAfter optimization energy is %lld", gc->compute_energy());
		e = gc->compute_energy();
//printf("\n%lld", e);

		for (int i = 0; i < num_pixels; i++)
			result[i] = gc->whatLabel(i);

		delete gc;

	} catch (GCException e) {
		e.Report();
	}

	//delete[] result;
	//delete[] smooth;
	delete[] data;

	Result *rEnd = new Result;

	rEnd->labels = result;
	rEnd->e = e;

	return rEnd;

}

void createImageLabelsShow(int *labels, int k, Mat &img) {

	unsigned char pv[k];

	for (int i = 0, j = 0; i < 256; i += (256 / k)) {
		pv[j] = i;
		j++;
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<unsigned char>(i, j) = pv[labels[i * img.cols + j]];
		}
	}

}

void createImageLabels(int *labels, int k, Mat &img) {

	unsigned char pv[k];

	for (int i = 0, j = 0; i < 256; i += (256 / k)) {
		pv[j] = i;
		j++;
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<unsigned char>(i, j) = labels[i * img.cols + j];
		}
	}

}

void writeCSV(string filename, cv::Mat m) {
//cv::Formatter const * c_formatter(cv::Formatter::get(cv::Formatter::FMT_CSV));
	ofstream myfile;
	myfile.open(filename.c_str(), ios::out);
	myfile << cv::format(m, cv::Formatter::FMT_CSV);//c_formatter->format(m);
	myfile.close();
}


int main(int argc, char **argv) {

	boost::log::core::get()->set_logging_enabled(false);

	int iflag = 0, oflag = 0, gflag = 0, dataflag = 0;
	char *cvalue = NULL;
	char *gvalue = NULL;
	char *outValue = NULL;
	char *dataValue = NULL;


	int index;
	int c;

	opterr = 0;

	while ((c = getopt(argc, argv, "o:g:i:d:")) != -1) {
		switch (c) {
		case 'i':
			cvalue = optarg;
			iflag = 1;
			break;
		case 'g':
			gvalue = optarg;
			gflag = 1;
			break;
		case 'o':
			outValue = optarg;
			oflag = 1;
			break;
		case 'd':
			dataValue = optarg;
			dataflag = 1;
			break;
		case '?':
			if (optopt == 'i')
				fprintf(stderr, "Option -i requires an argument.\n");
			else if (optopt == 'o')
				fprintf(stderr, "Option -o requires an argument.\n");
			else if (optopt == 'g')
				fprintf(stderr, "Option -g requires an argument.\n");
			else if (isprint(optopt))
				fprintf(stderr, "Unknown option `-%c'.\n", optopt);
			else
				fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
			return 1;
		default:
			abort();
		}
	}

	//printf("aflag = %d, bflag = %d, cvalue = %s\n", aflag, bflag, cvalue);

	for (index = optind; index < argc; index++) {
		printf("Non-option argument %s\n", argv[index]);
	}

	if (!iflag) {
		printf("-i require");
		abort();
	} else if (!oflag) {
		printf("-o require");
		abort();
	}
	if (!gflag) {
		printf("-g require");
		abort();
	}
	if (!dataflag) {
			printf("-d require");
			abort();
		}

	//BOOST_LOG_TRIVIAL(trace)<< "A trace severity message";
	//BOOST_LOG_TRIVIAL(debug)<< "A debug severity message";
	//BOOST_LOG_TRIVIAL(info)<< "An informational severity message";
	//BOOST_LOG_TRIVIAL(warning)<< "A warning severity message";
	//BOOST_LOG_TRIVIAL(error)<< "An error severity message";
	//BOOST_LOG_TRIVIAL(fatal)<< "A fatal severity message";

	//Load Data Term
	boost::filesystem::path dataTermPath(dataValue);
	boost::filesystem::path dataTermName("dataEnergy.cimg");
	boost::filesystem::path fullDataTermPath = dataTermPath / dataTermName;

	CImg<float> dataTerm(fullDataTermPath.string().c_str());

	//printf("\nData Term %f\n", dataTerm(0,0,0,0));


	string fileNameGroundTruth(gvalue);
	BOOST_LOG_TRIVIAL(info)<< "Load groundtruth " << fileNameGroundTruth;
	IcgBench::IcgBenchFileIO groudTruth(fileNameGroundTruth);

	boost::filesystem::path fileImagePath(cvalue);
	boost::filesystem::path fileImage(groudTruth.getFileName());
	boost::filesystem::path fullImagePath = fileImagePath / fileImage;
	BOOST_LOG_TRIVIAL(info)<< "Load Image " << fullImagePath;
	Mat grayImage, labelSuperPixels;
	int numSuperPixels = loadImage(fullImagePath.string(), grayImage, labelSuperPixels);
	if (numSuperPixels) {

		int k = groudTruth.getNumLabels();
		Result *r = NULL;
		int s_labels[k];
		int c_labels[k];
		for (int i = 0; i < k; ++i) {
			c_labels[i] = 0;
			s_labels[i] = 0;
		}

		vector<vector<IcgBench::Seed>> super(numSuperPixels);
		for (int i = 0; i < super.size(); ++i) {
			vector<IcgBench::Seed> s;
			super[i] = s;
		}

		for(int i = 0; i < labelSuperPixels.size().height; i++){
			for(int j = 0; j < labelSuperPixels.size().width; j++){
				IcgBench::Seed s;
				s.x = i;
				s.y = j;
				s.label = i * grayImage.cols + j;//labelSuperPixels.at<unsigned int>(i,j);
				super[labelSuperPixels.at<unsigned int>(i,j)].push_back(s);
			}
		}

		BOOST_LOG_TRIVIAL(info)<< "Define seeds using ground truth ";
		std::vector<IcgBench::Seed> seeds = groudTruth.getSeeds();
		for (unsigned int i = 0; i < seeds.size(); ++i) {
			IcgBench::Seed seed = seeds[i];
			//if ( seed.x <= grayImage.cols &&  seed.y <= grayImage.rows)
			unsigned int p = grayImage.at<unsigned char>(seed.y, seed.x);
			s_labels[seed.label] += p;
			c_labels[seed.label]++;
		}

		for (int i = 0; i < k; ++i) {
			s_labels[i] /= c_labels[i];
		}

		BOOST_LOG_TRIVIAL(info)<< "Execute segmentation";
		double t = (double) getTickCount();

		r = GridGraph_DMAT(grayImage, k, s_labels, seeds, c_labels, &dataTerm, super);

		t = ((double) getTickCount() - t) / getTickFrequency();

		BOOST_LOG_TRIVIAL(info)<< "Show result";

		Mat seg(grayImage.rows, grayImage.cols, CV_8SC1);
		createImageLabelsShow(r->labels, k - 1, seg);

		//imshow("Seg", seg);
		//waitKey(0);

		BOOST_LOG_TRIVIAL(info)<< "Save Label and Seed path";

		Mat labelsMat(grayImage.rows, grayImage.cols, CV_8SC1);
		Mat seedsMat = Mat::ones(grayImage.rows, grayImage.cols, CV_8SC1) * -1;

		boost::filesystem::path fileNameGroundTruthPath(fileNameGroundTruth);
		boost::filesystem::path fileOutPath(outValue);
		boost::filesystem::path fullOutPath = fileOutPath
				/ fileNameGroundTruthPath.stem();
		boost::filesystem::create_directory(fullOutPath);
		boost::filesystem::path extensionLabel(".label");
		boost::filesystem::path extensionSeed(".seed");

		boost::filesystem::path fullOutLabelPath("gd.label");
		//fullOutLabelPath.replace_extension(extensionLabel);
		fullOutLabelPath = fullOutPath / fullOutLabelPath;
		boost::filesystem::path fullOutSeedPath("gd.seed");// = fullOutPath;
		fullOutSeedPath = fullOutPath / fullOutSeedPath;
		//fullOutSeedPath.replace_extension(extensionSeed);

		createImageLabels(r->labels, k - 1, labelsMat);

//Label unsigned
		int sizeLabels = grayImage.rows * grayImage.cols;
		unsigned int labels[grayImage.rows * grayImage.cols];
		for (int i = 0; i < sizeLabels; ++i) {
			labels[i] = r->labels[i];
		}

		IcgBench::LabelImage labelImageIcg(labels, grayImage.cols,
				grayImage.rows);

		IcgBench::LabelImage* a = groudTruth.getLabels();
		IcgBench::LabelImage* b = &labelImageIcg;

//printf("%d\t%d\t%d\t%d\n", a->height(), b->height(), a->width(), b->width());

		double score = IcgBench::computeDiceScore(*a, *b);

		printf("%lld\t%.6lf\t%d\t%d\t%3i\n", r->e, score, numSuperPixels, k, (int) (t * 1000));

		writeCSV(fullOutLabelPath.string(), labelsMat);
		writeCSV(fullOutSeedPath.string(), seedsMat);

		BOOST_LOG_TRIVIAL(info)<< "End";

		return 0;

	} else {
		fprintf(stderr, "Not load image\n");
		abort();
	}



	//Load Image
	/*Mat gray;
	 int *labels = NULL;
	 Result *r;
	 int k  = 3;
	 if (loadImage(cvalue, gray)) {
	 //imshow("Gray image", gray);



	 long long eBefore = 0;
	 int l = 20;

	 for (k = 4; k < 5; k++) {

	 l = k;
	 int f_labels[l];

	 for (int i = 0, j = 0; i < 256; i += (256 / l)) {
	 f_labels[j] = i;
	 j++;
	 }

	 r = GridGraph_DMAT(gray, l, f_labels);

	 if ( k == 2 ){
	 eBefore = r->e;
	 labels = r->labels;

	 }

	 if (r->e >= eBefore ){
	 labels = r->labels;
	 break;
	 }

	 cout <<"E=" << r->e << " k = " << k << endl;

	 eBefore = r->e;


	 delete[] r->labels;
	 delete r;

	 }


	 //
	 waitKey(0);

	 }

	 Mat seg(gray.rows, gray.cols, CV_8SC1);

	 createImageLabelsShow(labels, k-1, seg);

	 Mat labelsMat(gray.rows, gray.cols, CV_8SC1);
	 Mat seedsMat = Mat::ones(gray.rows, gray.cols, CV_8SC1)*-1;

	 createImageLabels(labels, k-1, labelsMat);

	 cout << labelsMat << endl;


	 writeCSV("teste.label", labelsMat);
	 writeCSV("teste.seed", seedsMat);


	 /*for (int i = 0; i < gray.rows; i++) {
	 for (int j = 0; j < gray.cols; j++) {
	 printf("%d ", labels[i * gray.cols + j]);

	 }
	 printf("\n");
	 }*/

	//Mat seg(gray.rows, gray.cols, CV_32SC1, labels);
	//Mat cm_img0;
	// Apply the colormap:
	//applyColorMap(seg, cm_img0, COLORMAP_HOT);
	// Show the result:
	//imshow("Seg", seg);
	//waitKey(0);
	//return 0;
	/*int width = 10;
	 int height = 5;
	 int num_pixels = width*height;
	 int num_labels = 7;


	 // smoothness and data costs are set up one by one, individually
	 GridGraph_Individually(width,height,num_pixels,num_labels);

	 // smoothness and data costs are set up using arrays
	 GridGraph_DArraySArray(width,height,num_pixels,num_labels);

	 // smoothness and data costs are set up using functions
	 GridGraph_DfnSfn(width,height,num_pixels,num_labels);

	 // smoothness and data costs are set up using arrays.
	 // spatially varying terms are present
	 GridGraph_DArraySArraySpatVarying(width,height,num_pixels,num_labels);

	 //Will pretend our graph is
	 //general, and set up a neighborhood system
	 // which actually is a grid
	 GeneralGraph_DArraySArray(width,height,num_pixels,num_labels);

	 //Will pretend our graph is general, and set up a neighborhood system
	 // which actually is a grid. Also uses spatially varying terms
	 GeneralGraph_DArraySArraySpatVarying(width,height,num_pixels,num_labels);

	 printf("\n  Finished %d (%d) clock per sec %d",clock()/CLOCKS_PER_SEC,clock(),CLOCKS_PER_SEC);
	 */




}

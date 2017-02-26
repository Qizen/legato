#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#define IMAGE_NAME "BSDS_62.jpg"
//#define IMAGE_NAME "image3.jpg"
#define PI 3.14159265
#define DEBUG 0

using namespace cv;
using namespace std;

double ratio = 3;

void trackCallback(int trackerValue, void* userData)
{
	Mat* img = static_cast<Mat*>(userData);
	Mat edges;
	edges.create(img->size(), img->type());
	//Mat* foo = new Mat(image.size(), image.type());
	//cvCvtColor(&image, &img_grey, CV_BGR2GRAY);
	Canny(*img, edges, trackerValue, trackerValue*ratio);
	imshow("Canny", edges);

	Mat blurCanny, blurred;
	blurCanny.create(img->size(), img->type());
	GaussianBlur(*img, blurred, Size(5, 5), 0);
	Canny(blurred, blurCanny, trackerValue, trackerValue*ratio);
	imshow("Canny Blurred", blurCanny);
}

void sobel(Mat& inputImage, Mat& outputX, Mat& outputY)
{
	//TODO: add image border handling
	outputX.create(inputImage.size(), CV_32F);
	outputY.create(inputImage.size(), CV_32F);

	for (int y = 1; y < inputImage.rows - 1; y++)
	{
		for (int x = 1; x < inputImage.cols - 1; x++)
		{
			float valX = 0;
			valX = (-1 * inputImage.at<uchar>(y - 1, x - 1)) + inputImage.at<uchar>(y - 1, x + 1)
				+ (-2 * inputImage.at<uchar>(y, x - 1)) + 2 * inputImage.at<uchar>(y, x + 1)
				+ (-1 * inputImage.at<uchar>(y + 1, x - 1)) + inputImage.at<uchar>(y + 1, x + 1);

			float valY = 0;
			valY = inputImage.at<uchar>(y - 1, x - 1) + 2 * inputImage.at<uchar>(y - 1, x) + inputImage.at<uchar>(y - 1, x + 1)
				+ (-1 * inputImage.at<uchar>(y + 1, x - 1)) + (-2 * inputImage.at<uchar>(y + 1, x)) + (-1 * inputImage.at<uchar>(y + 1, x + 1));
			outputX.at<float>(y, x) = valX;
			outputY.at<float>(y, x) = valY;
		}
	}
}

void sobelCombineXY(Mat& inputX, Mat& inputY, Mat& output)
{
	//this also needs to handle image borders (start at 0 and end at m.rows() etc, rather than one off)
	output.create(inputX.size(), CV_8U);
	for (int y = 1; y < inputX.rows - 1; y++)
	{
		for (int x = 1; x < inputX.cols - 1; x++)
		{
			float foo = sqrt(pow(inputX.at<float>(y, x), 2) + pow(inputY.at<float>(y, x), 2)) * 255 / 360;
			output.at<uchar>(y, x) = foo;
			//cout << foo<<endl;
		}
	}
}

void sobelOrientations(Mat& inputX, Mat& inputY, Mat& output)
{
	output.create(inputX.size(), CV_8U);
	for (int y = 1; y < output.rows - 1; y++)
	{
		for (int x = 1; x < output.cols - 1; x++)
		{
			//result between 0 and 2 Pi
			 float rads = atan2(inputY.at<float>(y, x), inputX.at<float>(y, x)) + PI;

			 //for HSV, openCV has H in range 0-179
			 uchar degrees = (rads * 90) / (PI);
			 output.at<uchar>(y, x) = degrees;
			//printf("%f\n", atan(inputY.at<float>(y, x) / inputX.at<float>(y, x)));
			//printf("%i\n", output.at<uchar>(y, x));
		}
	}
}

void maxNotVisited(Mat& local, Mat&prevVisited, int offsetX, int offsetY, int& maxVal, Point& loc)
{
	Mat mask;
	mask.create(Size(3, 3), CV_32S);
	mask = mask.zeros(Size(3, 3), CV_32S);

	//will crash if you give it a border pixel
	for (int y = offsetY - 1; y <= offsetY + 1; y++)
	{
		for (int x = offsetX - 1; x <= offsetX + 1; x++)
		{
			if (prevVisited.at<int>(y, x) == 0)
			{
				mask.at<int>(y - (offsetY - 1), x - (offsetX - 1)) = 1;
			}
		}
	}

	Mat temp;
	temp.create(Size(3, 3), CV_32S);
	temp = mask.mul(local);
	double val;
	minMaxLoc(temp, NULL, &val, NULL, &loc);
	maxVal = (int)val;
}

//todo move this
#define INITIAL_EDGE_THRESHOLD 50

float f0[3][3] = { { 1.3, 1.5, 1.3 },
{ 1.0, 1.0, 1.0 },
{ 1.3, 1.5, 1.3 } };
Mat w0(Size(3, 3), CV_32F, &f0);

float f1[3][3] = { { 1.0, 1.3, 1.5 },
{ 1.3, 1.0, 1.3 },
{ 1.5, 1.3, 1.0 } };
Mat w1(Size(3, 3), CV_32F, &f1);

float f2[3][3] = { { 1.3, 1.0, 1.3 },
{ 1.5, 1.0, 1.5 },
{ 1.3, 1.0, 1.3 } };
Mat w2(Size(3, 3), CV_32F, &f2);

float f3[3][3] = { { 1.5, 1.3, 1.0 },
{ 1.3, 1.0, 1.3 },
{ 1.0, 1.3, 1.5 } };
Mat w3(Size(3, 3), CV_32F, &f3);

int pixInEdgeCount = 0;

// inoutMat is uchar, weights is float
void weightMat(Mat& inoutMat, Mat& weights)
{
	for (int y = 0; y < inoutMat.rows; y++)
	{
		for (int x = 0; x < inoutMat.cols; x++)
		{
			inoutMat.at<int>(y, x) = inoutMat.at<int>(y, x) * weights.at<float>(y, x);
		}
	}
}


void processPixel(Mat* inputEdges, Mat* inputOrientations, Mat* prevVisited, int offX, int offY, int edgeNum, Mat* out)
{
	if (DEBUG)
	{
		namedWindow("out", WINDOW_NORMAL);
		imshow("out", *out);
		waitKey(1);
	}

	//todo handle borders better
	if (offX <= 1 || offY <= 1 || offX >= inputEdges->cols - 1 || offY >= inputEdges->rows - 1)
		return;

	pixInEdgeCount++;

	if (DEBUG)
	{
		cout << "co-ord: " << offX << ", " << offY << " pixel: " << pixInEdgeCount << " in edge: " << edgeNum << endl;
	}
	prevVisited->at<int>(offY, offX) = edgeNum;
	out->at<uchar>(offY, offX) = 255;

	// copy the region
	Mat localUchar = (*inputEdges)(Rect(offX - 1, offY - 1, 3, 3)).clone();
	Mat local;
	localUchar.convertTo(local, CV_32S);

	// apply weighting depending on inputOrientations
	// NB orientation of 0 -> horizontal edge (i.e. the line of the edge is vertical)
	// orientation range 0-180
	uchar or = inputOrientations->at<uchar>(offY, offX);
	// orientations are unidirectional, edges are bidirectional
	or = or % 90;
	// |
	if (or < 11 || or >= 79)
	{
		weightMat(local, w0);
	}
	// /
	else if (or > 56 && or < 79)
	{
		//local = local.mul(w1);
		weightMat(local, w1);
	}
	// --
	else if (or > 33 && or <= 56)
	{
		//local = local.mul(w2);
		weightMat(local, w2);
	}
	// \ 
	else
	{
		//local = local.mul(w3);
		weightMat(local, w3);
	}

	// decide here whether interior, corner, junction, end, isolated
	// do this pre or post weighting? maybe before actually -- don't want to unfairly ignore things for corners etc.

	// now we want to process the neighbours and follow the edge
	// for first hack, pick the strongest edge not yet visited and go
	Point loc;
	int max;
	maxNotVisited(local, *prevVisited, offX, offY, max, loc);
	if (max > INITIAL_EDGE_THRESHOLD)
	{
		processPixel(inputEdges, inputOrientations, prevVisited, offX + (loc.x - 1), offY + (loc.y - 1), edgeNum, out);
	}
}


void oldprocessPixel(Mat& inputEdges, Mat& inputOrientations, Mat& prevVisited, int offX, int offY, int edgeNum, Mat& out)
{
	namedWindow("out", WINDOW_NORMAL);
	imshow("out", out);
	waitKey(1);

	//todo handle borders better
	if (offX <= 1 || offY <= 1 || offX >= inputEdges.cols - 1 || offY >= inputEdges.rows - 1)
		return;

	pixInEdgeCount++;
	cout << "co-ord: " << offX << ", " << offY << " pixel: " << pixInEdgeCount << " in edge: " << edgeNum << endl;

	prevVisited.at<float>(offY, offX) = edgeNum;
	out.at<uchar>(offY, offX) = 255;

	// copy the region
	Mat local = inputEdges(Rect(offX - 1, offY - 1, 3, 3)).clone();

	// apply weighting depending on inputOrientations
	// NB orientation of 0 -> horizontal edge (i.e. the line of the edge is vertical)
	// orientation range 0-180
	if (inputOrientations.at<uchar>(offY, offX) < 23 || inputOrientations.at<uchar>(offY, offX) > 158)
	{
		//local = local.mul(w0);
		weightMat(local, w0);
	}
	else if (inputOrientations.at<uchar>(offY, offX) < 68)
	{
		//local = local.mul(w1);
		weightMat(local, w1);
	}
	else if (inputOrientations.at<uchar>(offY, offX) < 113)
	{
		weightMat(local, w2);
		//local = local.mul(w2);
	}
	else
	{
		weightMat(local, w3);
		//local = local.mul(w3);
	}

	// decide here whether interior, corner, junction, end, isolated
	// do this pre or post weighting? maybe before actually -- don't want to unfairly ignore things for corners etc.

	// now we want to process the neighbours and follow the edge
	// for first hack, pick the strongest edge not yet visited and go
	Point loc;
	int max;
	maxNotVisited(local, prevVisited, offX, offY, max, loc);
	if (max > INITIAL_EDGE_THRESHOLD)
	{
		processPixel(&inputEdges, &inputOrientations, &prevVisited, offX + (loc.x - 1), offY + (loc.y - 1), edgeNum, &out);
	}
}


void pathfind(Mat& inputEdges, Mat& inputOrientations, Mat& output)
{
	// Q: should we do non-maxima suppression style things first?

	// ==== Algo Outline ====
	// Find a "strong" edge
	// Weight its surroundings (3x3 at first) based on orientation (rotated 90 degs)
	// Determine if inside, endpoint, corner, junction, or isolated based on neighbours
	// Record this somehow
	// Record that we have visited this pixel
	// Store computed orientation data -- this maybe in stack kinda vectory thing -- we need to know the specific sequence of orientation for this particular curve.
	// Go to strongest neighbour(s)
	// Repeat, using a combination of og orientation and computed orientation to determine weighting
	// Have limiting things to factor in max change etc? I guess they're implicitly handled in corner, endpoint, junction checks.
	Mat prevVisited;
	prevVisited.create(inputEdges.size(), CV_32S);
	prevVisited = prevVisited.zeros(inputEdges.size(), CV_32S);

	Mat out;
	out.create(inputEdges.size(), CV_8U);
	out = out.zeros(inputEdges.size(), CV_8U);

	int edgeNum = 1;

	//TODO: border processing
	for (int y = 1; y < inputEdges.rows - 1; y++)
	{
		for (int x = 1; x < inputEdges.cols - 1; x++)
		{
			if (inputEdges.at<uchar>(y, x) > INITIAL_EDGE_THRESHOLD)
			{
				if (!prevVisited.at<int>(y, x))
				{
					if (DEBUG)
					{
						cout << "processing edge: " << (int)edgeNum << endl;
					}

					prevVisited.at<int>(y, x) = edgeNum;
					out.at<uchar>(y, x) = 255;
					// copy the region
					Mat localUchar = inputEdges(Rect(x - 1, y - 1, 3, 3)).clone();
					Mat local;
					localUchar.convertTo(local, CV_32S);

					// apply weighting depending on inputOrientations
					// NB orientation of 0 -> horizontal edge (i.e. the line of the edge is vertical)
					// orientation range 0-180
					uchar or = inputOrientations.at<uchar>(y, x);
					// orientations are unidirectional, edges are bidirectional
					or = or % 90;
					// |
					if (or < 11 || or >= 79)
					{
						weightMat(local, w0);
					}
					// /
					else if (or > 56 && or < 79)
					{
						//local = local.mul(w1);
						weightMat(local, w1);
					}
					// --
					else if (or > 33 && or <= 56)
					{
						//local = local.mul(w2);
						weightMat(local, w2);
					}
					// \ 
					else
					{
						//local = local.mul(w3);
						weightMat(local, w3);
					}
					
					// decide here whether interior, corner, junction, end, isolated
					// do this pre or post weighting? maybe before actually -- don't want to unfairly ignore things for corners etc.

					// now we want to process the neighbours and follow the edge
					// for first hack, pick the strongest edge not yet visited and go
					Point loc;
					int max;
					maxNotVisited(local, prevVisited, x, y, max, loc);
					if (max > INITIAL_EDGE_THRESHOLD)
					{
						processPixel(&inputEdges, &inputOrientations, &prevVisited, x + (loc.x - 1), y + (loc.y - 1), edgeNum, &out);
					}
					edgeNum++;
					pixInEdgeCount = 0;
				}
			}
		}
	}

	// output prevVisited for now -- should be a pic with each edge being a different shade of grey
	output = out;
}

void printMat(Mat& m)
{
	for (int y = 0; y < m.rows; y++)
	{
		for (int x = 0; x < m.cols; x++)
		{
			cout << m.at<float>(y, x) << " ";
		}
		cout << endl;
	}
}

int main(int argc, char** argv)
{
	//just testing array assignment
	float arr[3][3] = { {1, 1, 1}
					, {2, 2, 2}
					, {3, 3, 3} };
	Mat foo(Size(3, 3), CV_32F, &arr);
	printMat(foo);
	foo = foo.mul(foo);
	printMat(foo);
	Mat image;
	image = imread(IMAGE_NAME, IMREAD_GRAYSCALE); // Read the file
	if (image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	Mat resized_image = image;
	//resize(image, resized_image, Size(), 0.2, 0.2, 1);
	namedWindow("Display window", WINDOW_NORMAL); // Create a window for display.
	imshow("Display window", resized_image); // Show our image inside it.

	createTrackbar("Low Threshold", "Display window", nullptr, 100, trackCallback, &resized_image);
	trackCallback(0, &resized_image);

	Mat outputX, outputY;
	Mat blurred;
	blur(resized_image, blurred, Size(3, 3));

	//TODO: Blur currently turned off, turn back on?
	sobel(resized_image, outputX, outputY);
	imshow("Sobel X", outputX);
	imshow("Sobel Y", outputY);
	
	Mat s;
	sobelCombineXY(outputX, outputY, s);
	imshow("Sobel result", s);
	
	Mat orientations;
	sobelOrientations(outputX, outputY, orientations);
	imshow("orientations", orientations);

	Mat colourOrientations;
	colourOrientations.create(orientations.size(), CV_8UC3);
	
	for (int y = 1; y < colourOrientations.rows - 1; y++)
	{
		for (int x = 1; x < colourOrientations.cols - 1; x++)
		{
			colourOrientations.at<Vec3b>(y, x) = Vec3b(orientations.at<uchar>(y, x), 255, s.at<uchar>(y, x));
		}
	}

	Mat bgr_orientations;
	cvtColor(colourOrientations, bgr_orientations, CV_HSV2BGR);

	imshow("orientations as colours", bgr_orientations);
	Mat ogSobel;
	Sobel(resized_image, ogSobel, -1, 1, 0);
	imshow("Their Sobel X", ogSobel);


	namedWindow("out", WINDOW_NORMAL);
	waitKey(0);

	Mat legato;
	pathfind(s, orientations, legato);
	namedWindow("legato", WINDOW_NORMAL);
	imshow("legato", legato);

	waitKey(0);
	return 0;
}
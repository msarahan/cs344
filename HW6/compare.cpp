#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils.h"

void showImages(cv::Mat refImage, cv::Mat userImage)
{
	cv::Mat diffImage=refImage.clone();
	cv::subtract(refImage,userImage,diffImage);
	cv::namedWindow( "Your Image", CV_WINDOW_AUTOSIZE );// Create a window for display.
	cv::imshow( "Your Image", userImage );                   // Show our image inside it.
	if((int)(cv::sum(diffImage)[0])>0)
	{
		// show the reference and difference images if the user's image did not pass the test.
		cv::namedWindow( "Reference Image", CV_WINDOW_AUTOSIZE );// Create a window for display.
		cv::imshow( "Reference Image", refImage );                   // Show reference image inside it.
		cv::namedWindow( "Difference Image", CV_WINDOW_AUTOSIZE );// Create a window for display.
		cv::imshow( "Difference Image", diffImage );                   // Show difference image inside it.
	}
	else
	{
		std::cout << "PASS" << std::endl;
	}
	cv::waitKey(0);                                          // Wait for a keystroke in the window
}

void compareImages(std::string reference_filename, std::string test_filename, bool useEpsCheck,
				   double perPixelError, double globalError)
{
  cv::Mat reference = cv::imread(reference_filename, -1);
  cv::Mat test = cv::imread(test_filename, -1);

  cv::Mat diff = abs(reference - test);

  showImages(reference,test);

  cv::Mat diffSingleChannel = diff.reshape(1, 0); //convert to 1 channel, same # rows

  double minVal, maxVal;

  cv::minMaxLoc(diffSingleChannel, &minVal, &maxVal, NULL, NULL); //NULL because we don't care about location

  //now perform transform so that we bump values to the full range

  diffSingleChannel = (diffSingleChannel - minVal) * (255. / (maxVal - minVal));

  diff = diffSingleChannel.reshape(reference.channels(), 0);

  cv::imwrite("HW6_differenceImage.png", diff);
  //OK, now we can start comparing values...
  unsigned char *referencePtr = reference.ptr<unsigned char>(0);
  unsigned char *testPtr = test.ptr<unsigned char>(0);

  if (useEpsCheck) {
    checkResultsEps(referencePtr, testPtr, reference.rows * reference.cols * reference.channels(), perPixelError, globalError);
  }
  else
  {
    checkResultsExact(referencePtr, testPtr, reference.rows * reference.cols * reference.channels());
  }

  std::cout << "PASS" << std::endl;
  return;
}

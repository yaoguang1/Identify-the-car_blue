#include "stdafx.h"
#include "cv.h"
#include "highgui.h"
#include "cxcore.h"
#include "omp.h"

using namespace cv;
using namespace std;

#define T_ANGLE_THRE 10
#define T_SIZE_THRE 10


void GetDiffImage(IplImage* src1, IplImage* src2, IplImage* dst, int nThre)   //得到不同的图像进行处理
{
	unsigned char* SrcData1 = (unsigned char*)src1->imageData;
	unsigned char* SrcData2 = (unsigned char*)src2->imageData;
	unsigned char* DstData = (unsigned char*)dst->imageData;
	int step = src1->widthStep / sizeof(unsigned char);

	omp_set_num_threads(8);   //设置线程数
#pragma omp parallel for

	for (int nI = 0; nI<src1->height; nI++)    //height图像高像素数
	{
		for (int nJ = 0; nJ <src1->width; nJ++)    //width图像宽像素数
		{
			if (SrcData1[nI*step + nJ] - SrcData2[nI*step + nJ]> nThre)
			{
				DstData[nI*step + nJ] = 255;
			}
			else
			{
				DstData[nI*step + nJ] = 0;
			}
		}
	}
}

vector<CvBox2D> ArmorDetect(vector<CvBox2D> vEllipse)    //装甲检测  //用opencv最小外接矩形去表示一个类椭圆形Ellipse的高度
{
	vector<CvBox2D> vRlt;
	CvBox2D Armor;
	int nL, nW;
	vRlt.clear();
	if (vEllipse.size() < 2)
		return vRlt;
	for (unsigned int nI = 0; nI < vEllipse.size() - 1; nI++)
	{
		for (unsigned int nJ = nI + 1; nJ < vEllipse.size(); nJ++)
		{
			if (abs(vEllipse[nI].angle - vEllipse[nJ].angle) < T_ANGLE_THRE && 
			    abs(vEllipse[nI].size.height - vEllipse[nJ].size.height) < (vEllipse[nI].size.height + vEllipse[nJ].size.height) / 10 && 
			    abs(vEllipse[nI].size.width - vEllipse[nJ].size.width) < (vEllipse[nI].size.width + vEllipse[nJ].size.width)/10)
			{
				Armor.center.x = (vEllipse[nI].center.x + vEllipse[nJ].center.x) / 2;
				Armor.center.y = (vEllipse[nI].center.y + vEllipse[nJ].center.y) / 2;
				Armor.angle = (vEllipse[nI].angle + vEllipse[nJ].angle) / 2;
				nL = (vEllipse[nI].size.height + vEllipse[nJ].size.height) / 2;
				nW = sqrt((vEllipse[nI].center.x - vEllipse[nJ].center.x) * (vEllipse[nI].center.x - vEllipse[nJ].center.x) + (vEllipse[nI].center.y - vEllipse[nJ].center.y) * (vEllipse[nI].center.y - vEllipse[nJ].center.y));
				if (nL < nW)
				{
					Armor.size.height = nL;
					Armor.size.width = nW;
				}
				else
				{
					Armor.size.height = nW;
					Armor.size.width = nL;
				}
				vRlt.push_back(Armor);
			}
		}
	}
	return vRlt;
}

void DrawBox(CvBox2D box, IplImage* img)    //对给定的2D点集，寻找最小面积的包围矩形，
				       	    //使用函数 
					    //CvBox2D cvMinAreaRect2( const CvArr* points, CvMemStorage* storage=NULL ); 
{
	CvPoint2D32f point[4];
	int i;
	for (i = 0; i<4; i++)
	{
		point[i].x = 0;
		point[i].y = 0;
	}
	cvBoxPoints(box, point);   //用函数cvBoxPoints(box[count], point); 寻找盒子顶点 
	CvPoint pt[4];
	for (i = 0; i<4; i++)
	{
		pt[i].x = (int)point[i].x;
		pt[i].y = (int)point[i].y;
	}
	cvLine(img, pt[0], pt[1], CV_RGB(0, 0, 255), 2, 8, 0);
	cvLine(img, pt[1], pt[2], CV_RGB(0, 0, 255), 2, 8, 0);
	cvLine(img, pt[2], pt[3], CV_RGB(0, 0, 255), 2, 8, 0);
	cvLine(img, pt[3], pt[0], CV_RGB(0, 0, 255), 2, 8, 0);
}

int main()
{
	CvCapture* pCapture0 = cvCreateFileCapture("RawImage\\BlueCar.avi");   //highgui.h的CVCreateFileCapture函数-视频读写
	//CvCapture* pCapture0 = cvCreateCameraCapture(0);
	IplImage* pFrame0 = NULL;   //使用IplImage*数据结构来表示图像
	CvSize pImgSize;
	CvScalar sColour;   //cxcore.h中函数，一般用来存放像素值（不一定是灰度值哦）的，最多可以存放4个通道的。
	CvBox2D s;
	vector<CvBox2D> vEllipse;
	vector<CvBox2D> vRlt;
	vector<CvBox2D> vArmor;
	CvScalar sl;
	bool bFlag = false;
	CvSeq *pContour = NULL;   //CvSeq本身就是一个可增长的序列

	pFrame0 = cvQueryFrame(pCapture0);    //函数cvQueryFrame从摄像头或者文件中抓取一帧，然后解压并返回这一帧。

	pImgSize = cvGetSize(pFrame0);   //cvGetSize是 OpenCV提供的一种操作矩阵图像的函数。得到二维的数组的尺寸,以CvSize返回。

	IplImage *pRawImg = cvCreateImage(pImgSize, IPL_DEPTH_8U, 3);   //cvCreateImage也就是创建图像,并进行初始化设置。
									//像素的位深度: IPL_DEPTH_8U(8位无符号整数).
									//例如：创建一个宽为360,高为640的3通道图像（RGB图像）
									//IplImage* img=cvCreateImage( cvSize(360,640), IPL_DEPTH_8U,3 );
	IplImage* pHImage = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);
	IplImage* pRImage = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);
	IplImage* pGImage = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);
	IplImage *pBImage = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);
	IplImage *pBinary = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);
	IplImage *pRlt = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);

	CvSeq* lines = NULL;
	/*opencv之内存存储器——CvMemStorage与CvSeq
	CvMemStorage *storage=cvCreateMemStorage(block_size);
	用来创建一个内存存储器，来统一管理各种动态对象的内存。
	函数返回一个新创建的内存存储器指针。
	参数block_size对应内存器中每个内存块的大小，为0时内存块默认大小为64k。*/
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvMemStorage* pStorage = cvCreateMemStorage(0);  //轮廓的存储容器
	while (1)
	{
		if (pFrame0)    //有视频帧输入
		{
			/* void cvSplit(const CvArr* src,CvArr *dst0,CvArr *dst1, CvArr *dst2, CvArr *dst3);  
			有些时候处理多通道图像时不是很方便，在这种情况下，可以利用cvSplit（）分别复制每个通道到多个单通道图像，
			如果需要，cvSplit（）函数将复制src（即 源 多通道图像）的各个通道到图像dst0、dst1、dst2、dst3中。
			目标图像必须与源图像在大小和数据类型上匹配，当然也应该是单通道的图像*/
			cvSplit(pFrame0, pBImage, pGImage, pRImage, 0);   //RGB三通道各一个
			GetDiffImage(pBImage, pRImage, pBinary, 90);
			cvDilate(pBinary, pHImage, NULL, 3);   //Dilate 先膨胀，可以使不连通的图像合并成块。(这两个形态学函数总是成对出现)
			cvErode(pHImage, pRlt, NULL, 1);  //Erode后腐蚀，可以消除较小独点如噪音。
			/* cvFindContours 在二值图像中寻找轮廓
			int cvFindContours(CvArr* image,   输入的8-比特、单通道图像.非零元素被当成1，0象素值保留为0-从而图像被看成二值的。
							CvMemStorage* storage,  得到的轮廓的存储容器 
							CvSeq** first_contour,  输出参数：包含第一个输出轮廓的指针
							int header_size=sizeof(CvContour), 如果 method=CV_CHAIN_CODE，则序列头的大小 >=sizeof(CvChain)，否则 >=sizeof(CvContour) . 
							int mode=CV_RETR_LIST,  提取模式.CV_RETR_CCOMP - 提取所有轮廓，并且将其组织为两层的 hierarchy: 顶层为连通域的外围边界，次层为洞的内层边界。
							int method=CV_CHAIN_APPROX_SIMPLE, 逼近方法,该方法即函数只保留末端的象素点;
							CvPoint offset=cvPoint(0,0)); 每一个轮廓点的偏移量. 当轮廓是从图像 ROI 中提取出来的时候，使用偏移量有用，因为可以从整个图像上下文来对轮廓做分析. 
							*/
			cvFindContours(pRlt, pStorage, &pContour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
			/*函数 cvFindContours 从二值图像中提取轮廓，并且返回提取轮廓的数目。
			指针 first_contour 的内容由函数填写。它包含第一个最外层轮廓的指针，如果指针为 NULL，
			则没有检测到轮廓（比如图像是全黑的）。*/
			for (; pContour != NULL; pContour = pContour->h_next)
			{
				if (pContour->total > 10)
				{
					bFlag = true;
					s = cvFitEllipse2(pContour);
					for (int nI = 0; nI < 5; nI++)
					{
						for (int nJ = 0; nJ < 5; nJ++)
						{
							if (s.center.y - 2 + nJ > 0 && s.center.y - 2 + nJ < 480 && s.center.x - 2 + nI > 0 && s.center.x - 2 + nI <  640)
							{
								sl = cvGet2D(pFrame0, (int)(s.center.y - 2 + nJ), (int)(s.center.x - 2 + nI));
								if (sl.val[0] < 200 || sl.val[1] < 200 || sl.val[2] < 200)
									bFlag = false;
							}
						}
					}
					if (bFlag)
					{
						vEllipse.push_back(s);
						//cvEllipseBox(pFrame0, s, CV_RGB(255, 0, 0), 2, 8, 0);
					}
				}
				
			}
			vRlt = ArmorDetect(vEllipse);   //检测装甲的块

			for (unsigned int nI = 0; nI < vRlt.size(); nI++)
				DrawBox(vRlt[nI], pFrame0);    //画出盒子


			//cvWriteFrame(writer, pRawImg);
			cvShowImage("Raw", pFrame0);
			cvWaitKey(0);
			vEllipse.clear();
			vRlt.clear();
			vArmor.clear();
		}
		pFrame0 = cvQueryFrame(pCapture0);
	}
	cvReleaseCapture(&pCapture0);
	return 0;
}

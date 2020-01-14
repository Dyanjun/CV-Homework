#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QImage>
#include <QFileDialog>
#include <vector>
#include <QString>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

#define THRESHOLD_VALUE 30     /* the threshold value of binary image */

using namespace std;
namespace Ui {
class Widget;
}

class Widget : public QWidget
{
    Q_OBJECT

public:
    explicit Widget(QWidget *parent = 0);
    ~Widget();

    QString filename;
    QImage photo;           /* the original image */
    cv::Mat preserved_img;
    cv::Mat pre_img;        /* Mat-version image before handled */
    cv::Mat color_img,img;  /* Mat-version gray/binary image */
    int kernal_size = 5;
    int** Belongsto;
    int NumberOfCaps;
    QString xyresult;
    vector<cv::Vec3f> circles1;
    vector<cv::Vec3f> circles;
private:
    Ui::Widget *ui;
    QMenuBar *menuBar = new QMenuBar(this);
    QMenu *menu;
    QAction *open;
    QAction *start;
    void display(cv::Mat *, int);
    void split();
    void binary_handler();
    void median_handler();
    void searchCircles();
    void searchSquares();
    void getCircle(vector<cv::Point> p, cv::Point* center, float* radius);
bool Widget::test(cv::Mat& src, int x, int y, int r);
public slots:
    void openfile();
    void handle();
};

#endif // WIDGET_H

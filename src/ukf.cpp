#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() 
{
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;

  n_aug_ = 7;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  is_initialized_ = false;

  time_us_ = 0;

  lambda_ = 3 - n_aug_;

  // init weights
  weights_ = VectorXd(2 * n_aug_+ 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < 2*n_aug_+1; ++i) 
    weights_(i) = 0.5 / (lambda_ + n_aug_);
  
  
}

UKF::~UKF() 
{
  // cout << "~ukf";
  // nis_radar_f_.close();
  // nis_laser_f_.close();
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) 
{
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) 
  {
    // init covariance matrix P
     // P_ << 1, 0, 0, 0, 0,
     //      0, 1, 0, 0, 0,
     //      0, 0, 1, 0, 0,
     //      0, 0, 0, 1, 0,
     //      0, 0, 0, 0, 1;

      P_ << std_radr_*std_radr_, 0, 0, 0, 0,
            0, std_radr_*std_radr_, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, std_radphi_, 0,
            0, 0, 0, 0, std_radphi_;

    // initialize the state x with the first measurement.
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) 
    {
      auto rho = meas_package.raw_measurements_[0];
      auto theta = meas_package.raw_measurements_[1];

      // v is different from rho_dot. initializing inital velocity v to zero
      x_ << rho*std::cos(theta), rho*std::sin(theta), 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }
    cout << "x: " << x_ << endl;

    // initialize timestamp
    time_us_ = meas_package.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;

    return;
  }


  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  // elapsed time (us to seconds)
  auto dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(dt);

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  // radar measurement
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) 
  {
    UpdateRadar(meas_package);
  }

  // laser measurement
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
  {
    UpdateLidar(meas_package);
  }

  // print the output
  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;

}

/**
 * Predicts sigma points Xsig_pred_, the state x_, and the state covariance matrix P_.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) 
{
  /*****************************************************************************
   *  augment x_ and P_ with process noise
   ****************************************************************************/

   //create augmented state
   VectorXd x_aug = VectorXd(n_aug_);
   x_aug.head(n_x_) = x_;
   x_aug(5) = 0;
   x_aug(6) = 0;

   //create augmented covariance matrix
   MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
   P_aug.fill(0.0);
   P_aug.topLeftCorner(n_x_, n_x_) = P_;
   P_aug(5,5) = std_a_ * std_a_;
   P_aug(6,6) = std_yawdd_ * std_yawdd_;


  /*****************************************************************************
   *  generate augmented sigma points Xsig_aug
   ****************************************************************************/

   MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
   Xsig_aug.col(0)  = x_aug;
   // L: sqrt of P_aug
   MatrixXd L = P_aug.llt().matrixL();
   for (int i = 1; i <= n_aug_; ++i)
   {
    Xsig_aug.col(i) = x_aug + std::sqrt(lambda_ + n_aug_) * L.col(i-1);
    Xsig_aug.col(i+n_aug_) = x_aug - std::sqrt(lambda_ + n_aug_) * L.col(i-1);
   }


  /*****************************************************************************
   *  sigma point prediction Xsig_pred__ (apply process model)
   ****************************************************************************/

  for (int i = 0; i < 2*n_aug_+1; ++i)
  {
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    // predicted state values
    double px_p, py_p;

    // handle case where object drives in a straight line
    // yawd = 0. avoid division by zero
    if (fabs(yawd) > 0.001) 
    {
        px_p = p_x + v/yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * (cos(yaw) - cos(yaw + yawd*delta_t));
    }
    else 
    {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    // add process noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;
    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    // update Xsig_pred_
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  /*****************************************************************************
   *  get predicted mean x_ and predicted covariance P_ from predicted 
      sigma points Xsig_pred_
   ****************************************************************************/

  // predicted state mean x_
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);

  // predicted state covariance matrix P_
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2.*M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.*M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }


}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) 
{
  /*****************************************************************************
   *  transform predicted sigma points Xsig_pred__ to measurement sigma points Zsig
   ****************************************************************************/
  int n_z = 2;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {
    // measurement model
    Zsig(0,i) = Xsig_pred_(0,i); // p_x
    Zsig(1,i) = Xsig_pred_(1,i); // p_y
  }

  /*****************************************************************************
   *  predict mean z_pred and covariance S from measurement sigma points Zsig
   ****************************************************************************/
  // mean predicted measurement z_pred
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
    z_pred = z_pred + weights_(i) * Zsig.col(i);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R <<    std_laspx_*std_laspx_, 0,
          0, std_laspy_*std_laspy_;
  S = S + R;


  /*****************************************************************************
   *  ukf state update x_ and P_
   ****************************************************************************/
  VectorXd z_meas = meas_package.raw_measurements_;

  // calculate cross correlation matrix Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = z_meas - z_pred;

  // update state mean x_ and covariance matrix P_
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();


  /*****************************************************************************
   *  nis
   ****************************************************************************/
  double nis = z_diff.transpose() * S.inverse() * z_diff;
  cout << "nis laser: " << nis << "\n";
  std::ofstream nis_laser_f_;
  nis_laser_f_.open("nis_laser.txt", std::ios_base::app);
  nis_laser_f_ << nis << "\n";
}

/**
 * Updates the state x_ and the state covariance matrix P_ using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) 
{
  /*****************************************************************************
   *  transform predicted sigma points Xsig_pred__ to measurement sigma points Zsig
   ****************************************************************************/
  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {
    // extract values
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }

  /*****************************************************************************
   *  predict mean z_pred and covariance S from measurement sigma points Zsig
   ****************************************************************************/
  // mean predicted measurement z_pred
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for(int i = 0; i < 2*n_aug_+1; i++)
    z_pred = z_pred + weights_(i) * Zsig.col(i);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R <<    std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0,std_radrd_*std_radrd_;
  S = S + R;


  /*****************************************************************************
   *  ukf state update x_ and P_
   ****************************************************************************/
  VectorXd z_meas = meas_package.raw_measurements_;

  // calculate cross correlation matrix Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = z_meas - z_pred;

  // angle normalization
  while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

  // update state mean x_ and covariance matrix P_
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();


  /*****************************************************************************
   *  nis
   ****************************************************************************/
  double nis = z_diff.transpose() * S.inverse() * z_diff;
  cout << "nis radar: " << nis << "\n";
  std::ofstream nis_radar_f_;
  nis_radar_f_.open("nis_radar.txt", std::ios_base::app);
  nis_radar_f_ << nis << "\n";

}

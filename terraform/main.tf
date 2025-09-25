provider "aws" {
  region = "us-east-1"
}

# S3 bucket for Dockerrun file
resource "aws_s3_bucket" "app_bucket" {
  bucket = "housing-app-tf"
}
# Upload Dockerrun
resource "aws_s3_object" "dockerrun" {
  bucket = aws_s3_bucket.app_bucket.id
  key    = "Dockerrun.aws.json"
  source = "../Dockerrun.aws.json"
  etag   = filemd5("../Dockerrun.aws.json")
}

# Attach ECR read-only policy to EB EC2 role
resource "aws_iam_role_policy_attachment" "ecr_readonly" {
  role       = "aws-elasticbeanstalk-ec2-role"
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

# Elastic Beanstalk Application
resource "aws_elastic_beanstalk_application" "housing_app" {
  name        = "housing-app-tf"
  description = "Housing API Application"
}

resource "aws_elastic_beanstalk_application_version" "app_version" {
  name        = "v-${timestamp()}"
  application = aws_elastic_beanstalk_application.housing_app.name
  bucket      = aws_s3_bucket.app_bucket.id
  key         = aws_s3_object.dockerrun.key
}

# Elastic Beanstalk Environment
resource "aws_elastic_beanstalk_environment" "housing_env" {
  name                = "Housing-app-env"
  application         = aws_elastic_beanstalk_application.housing_app.name
  # platform_arn = "arn:aws:elasticbeanstalk:us-east-1::platform/Docker running on 64bit Amazon Linux 2/4.3.1"
  solution_stack_name = "64bit Amazon Linux 2 v4.3.1 running Docker"
  version_label = aws_elastic_beanstalk_application_version.app_version.name
  wait_for_ready_timeout = "15m"

  # IAM instance profile
  setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "IamInstanceProfile"
    value     = "aws-elasticbeanstalk-ec2-role"
  }

  # app environment variable
  setting {
    namespace = "aws:elasticbeanstalk:application:environment"
    name      = "ENV"
    value     = "production"
  }

  setting {
    namespace = "aws:elasticbeanstalk:application"
    name      = "Application Healthcheck URL"
    value     = "/health"
  }

}

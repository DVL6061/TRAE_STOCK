# Outputs for AWS Infrastructure

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.stock_vpc.id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public_subnets[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private_subnets[*].id
}

output "load_balancer_dns" {
  description = "DNS name of the load balancer"
  value       = aws_lb.stock_alb.dns_name
}

output "load_balancer_zone_id" {
  description = "Zone ID of the load balancer"
  value       = aws_lb.stock_alb.zone_id
}

output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.stock_db.endpoint
  sensitive   = true
}

output "database_port" {
  description = "RDS instance port"
  value       = aws_db_instance.stock_db.port
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_cluster.stock_redis.cache_nodes[0].address
  sensitive   = true
}

output "redis_port" {
  description = "ElastiCache Redis port"
  value       = aws_elasticache_cluster.stock_redis.cache_nodes[0].port
}

output "autoscaling_group_name" {
  description = "Name of the Auto Scaling Group"
  value       = aws_autoscaling_group.stock_asg.name
}

output "security_group_ec2_id" {
  description = "ID of the EC2 security group"
  value       = aws_security_group.ec2_sg.id
}

output "security_group_alb_id" {
  description = "ID of the ALB security group"
  value       = aws_security_group.alb_sg.id
}

output "key_pair_name" {
  description = "Name of the EC2 Key Pair"
  value       = aws_key_pair.stock_key.key_name
}

output "application_url" {
  description = "URL to access the application"
  value       = "http://${aws_lb.stock_alb.dns_name}"
}

output "ssl_application_url" {
  description = "HTTPS URL to access the application (after SSL setup)"
  value       = "https://${var.domain_name}"
}

output "deployment_instructions" {
  description = "Instructions for completing the deployment"
  value = <<-EOT
    Deployment completed! Next steps:
    
    1. Point your domain '${var.domain_name}' to the load balancer:
       DNS Name: ${aws_lb.stock_alb.dns_name}
       Zone ID: ${aws_lb.stock_alb.zone_id}
    
    2. SSH into any instance using:
       ssh -i your-private-key.pem ubuntu@<instance-ip>
    
    3. The application will be available at:
       - HTTP: http://${aws_lb.stock_alb.dns_name}
       - HTTPS: https://${var.domain_name} (after SSL setup)
    
    4. Monitor the deployment:
       - Check Auto Scaling Group: ${aws_autoscaling_group.stock_asg.name}
       - View logs on instances
    
    5. Database connection details are available in the sensitive outputs.
  EOT
}
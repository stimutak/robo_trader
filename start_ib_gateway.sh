#!/bin/bash

# Set Java path
export JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home
export PATH=$JAVA_HOME/bin:$PATH

echo "Starting IB Client Portal Gateway..."
echo "=================================="
echo ""

cd clientportal.gw

# Start with specific Java options for compatibility
java \
-Djava.awt.headless=true \
-Dvertx.disableDnsResolver=true \
-Djava.net.preferIPv4Stack=true \
-Dvertx.logger-delegate-factory-class-name=io.vertx.core.logging.SLF4JLogDelegateFactory \
-cp "root:dist/ibgroup.web.core.iblink.router.clientportal.gw.jar:build/lib/runtime/*" \
ibgroup.web.core.clientportal.gw.GatewayStart \
--conf ../root/conf.yaml
@echo off
echo Creating completely fresh mobile app...

REM Navigate to a completely separate directory
cd C:\
mkdir TomatoMobileApp
cd TomatoMobileApp

REM Create new Expo app
npx create-expo-app@latest tomato-detector --template blank-typescript

REM Navigate to the new app
cd tomato-detector

echo Fresh mobile app created at C:\TomatoMobileApp\tomato-detector
echo Now run: cd C:\TomatoMobileApp\tomato-detector
echo Then run: npm install
pause


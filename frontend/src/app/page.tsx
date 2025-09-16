
"use client"

import type React from "react"
import { useState, useEffect } from "react" // Import useEffect
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card"
import { Button } from "../components/ui/button"
import { Input } from "../components/ui/input"
import { Label } from "../components/ui/label"
import { Alert, AlertDescription } from "../components/ui/alert"
import { Badge } from "../components/ui/badge"
import {
  Cloud,
  Sun,
  CloudRain,
  Thermometer,
  Droplets,
  Wind,
  Calendar,
  Bell,
  MapPin,
  Camera,
  Upload,
  X,
  CheckCircle,
  AlertTriangle,
  ArrowRight,
  Bug,
  Sprout,
  Leaf,
  Shield,
  TrendingUp,
} from "lucide-react"
import { useToast } from "../hooks/use-toast" // Import useToast

interface WeatherData {
  location: string
  current: {
    temp: number
    humidity: number
    windSpeed: number
    condition: string
    icon: string
  }
  forecast: Array<{
    date: string
    temp: { min: number; max: number }
    humidity: number
    condition: string
    icon: string
    precipitation: number
  }>
}


interface PlantingRecommendation {
  shouldPlant: boolean
  reason: string
  optimalDays: string[]
  tasks: Array<{
    task: string
    daysFromPlanting: number
    description: string
  }>
}

interface PredictionResult {
  pestName: string
  confidence: number
  description: string
  prevention: string[]
  pesticides: string[]
  images: string[]
}

export default function TomatoFarmingApp() {
  const [currentView, setCurrentView] = useState<"welcome" | "schedule" | "detection">("welcome")
  const [location, setLocation] = useState("")
  const [weatherData, setWeatherData] = useState<WeatherData | null>(null)
  const [recommendation, setRecommendation] = useState<PlantingRecommendation | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
  const [predictionLoading, setPredictionLoading] = useState(false)
  const [showCamera, setShowCamera] = useState(false)
  const [actualPlantingDate, setActualPlantingDate] = useState<string | null>(null) // New state for confirmed planting date

  const { toast } = useToast() // Initialize useToast

  const nigerianStates = [
    "Lagos",
    "Kano",
    "Kaduna",
    "Oyo",
    "Rivers",
    "Bayelsa",
    "Katsina",
    "Cross River",
    "Delta",
    "Sokoto",
    "Kebbi",
    "Osun",
    "Zamfara",
    "Gombe",
    "Kwara",
    "Niger",
    "Plateau",
    "Adamawa",
    "Imo",
    "Borno",
    "Ogun",
    "Taraba",
    "Yobe",
    "Edo",
    "Ondo",
    "Akwa Ibom",
    "Anambra",
    "Benue",
    "Jigawa",
    "Enugu",
    "Abia",
    "Bauchi",
    "Kogi",
    "Nasarawa",
    "Ebonyi",
    "FCT",
  ]

  const fetchWeatherData = async (locationName: string) => {
    setLoading(true)
    setError("")

    try {
      const response = await fetch(`/api/flask-weather?location=${encodeURIComponent(locationName)}`)
      if (!response.ok) {
        throw new Error(`Weather API responded with status: ${response.status}`)
      }
      const weatherData = await response.json()
      setWeatherData(weatherData)
      generatePlantingRecommendation(weatherData)
    } catch (err) {
      setError("Failed to fetch weather data. Please check your internet connection and try again.")
    } finally {
      setLoading(false)
    }
  }

  const generatePlantingRecommendation = (weather: WeatherData) => {
    const avgTemp =
      weather.forecast.reduce((sum, day) => sum + (day.temp.min + day.temp.max) / 2, 0) / weather.forecast.length
    const avgHumidity = weather.forecast.reduce((sum, day) => sum + day.humidity, 0) / weather.forecast.length
    const rainyDays = weather.forecast.filter((day) => day.precipitation > 50).length
    const shouldPlant = avgTemp >= 20 && avgTemp <= 35 && avgHumidity >= 60 && avgHumidity <= 80 && rainyDays <= 2

    const today = new Date()
    today.setHours(0, 0, 0, 0) // Normalize today's date to start of day

    const futureOptimalDays: string[] = []
    if (shouldPlant) {
      let daysFound = 0
      for (const day of weather.forecast) {
        const forecastDate = new Date(day.date)
        forecastDate.setHours(0, 0, 0, 0) // Normalize forecast date to start of day

        // Only consider dates that are today or in the future
        if (forecastDate >= today && daysFound < 3) {
          // Add some simple logic for "optimal" based on temperature and humidity
          if (day.temp.min >= 20 && day.temp.max <= 35 && day.humidity >= 60 && day.humidity <= 80) {
            futureOptimalDays.push(
              forecastDate.toLocaleDateString("en-US", { year: "numeric", month: "2-digit", day: "2-digit" }),
            )
            daysFound++
          }
        }
      }
    }
    setRecommendation({
      shouldPlant,
      reason: shouldPlant ? "Weather conditions are optimal for tomato planting" : "Weather conditions are not optimal",
      optimalDays: futureOptimalDays,
      tasks: [
        { task: "Watering", daysFromPlanting: 2, description: "Water seedlings gently in the morning" },
        { task: "Weeding", daysFromPlanting: 7, description: "Remove weeds around the plants" },
        { task: "Fertilizing", daysFromPlanting: 14, description: "Apply organic fertilizer" },
        { task: "Pruning", daysFromPlanting: 21, description: "Remove lower leaves and suckers" },
        { task: "Pest Check", daysFromPlanting: 28, description: "Inspect plants for pests and diseases" },
      ],
    })
  }

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      if (!file.type.match(/^image\/(jpeg|jpg|png)$/)) {
        setError("Please upload a valid image file (JPEG, JPG, or PNG)")
        return
      }
      if (file.size > 5 * 1024 * 1024) {
        setError("Image size should be less than 5MB")
        return
      }
      setSelectedImage(file)
      const reader = new FileReader()
      reader.onload = (e) => setImagePreview(e.target?.result as string)
      reader.readAsDataURL(file)
      setError("")
    }
  }

  const handleCameraCapture = () => {
    setShowCamera(true)
    setTimeout(() => {
      const canvas = document.createElement("canvas")
      canvas.width = 640
      canvas.height = 480
      const ctx = canvas.getContext("2d")
      if (ctx) {
        ctx.fillStyle = "#4ade80"
        ctx.fillRect(0, 0, canvas.width, canvas.height)
        ctx.fillStyle = "#166534"
        ctx.font = "24px Arial"
        ctx.fillText("Simulated Camera Capture", 150, 240)
        canvas.toBlob((blob) => {
          if (blob) {
            const file = new File([blob], "camera-capture.png", { type: "image/png" })
            setSelectedImage(file)
            setImagePreview(canvas.toDataURL())
            setShowCamera(false)
          }
        })
      }
    }, 2000)
  }

  const handlePestPrediction = async () => {
    if (!selectedImage) {
      setError("Please select or capture an image first")
      return
    }
    setPredictionLoading(true)
    setError("")

    try {
      const formData = new FormData()
      formData.append("image", selectedImage)
      if (location) formData.append("location", location)

      const response = await fetch("/api/flask-predict", { method: "POST", body: formData })
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Prediction failed")
      }
      const result = await response.json()
      setPredictionResult(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to analyze image. Please try again.")
    } finally {
      setPredictionLoading(false)
    }
  }

  const resetPrediction = () => {
    setSelectedImage(null)
    setImagePreview(null)
    setPredictionResult(null)
    setError("")
  }

  const resetToWelcome = () => {
    setCurrentView("welcome")
    setWeatherData(null)
    setRecommendation(null)
    resetPrediction()
    setLocation("")
    setError("")
    setActualPlantingDate(null) // Reset planting date
  }

  const getWeatherIcon = (condition: string) => {
    switch (condition.toLowerCase()) {
      case "sunny":
        return <Sun className="w-6 h-6 text-yellow-500" />
      case "partly cloudy":
        return <Cloud className="w-6 h-6 text-gray-500" />
      case "light rain":
      case "heavy rain":
        return <CloudRain className="w-6 h-6 text-blue-500" />
      default:
        return <Cloud className="w-6 h-6 text-gray-500" />
    }
  }

  // Function to handle confirming a planting date
  const handleConfirmPlanting = (date: string) => {
    setActualPlantingDate(date)
    toast({
      title: "Planting Date Confirmed!",
      description: `Your tomato planting is scheduled for ${date}. We'll send you reminders for tasks.`,
      variant: "success",
    })
  }

  // Effect to check for and trigger task reminders
  useEffect(() => {
    if (actualPlantingDate && recommendation?.tasks) {
      const plantingDate = new Date(actualPlantingDate)
      plantingDate.setHours(0, 0, 0, 0) // Normalize to start of day

      const today = new Date()
      today.setHours(0, 0, 0, 0) // Normalize today's date

      recommendation.tasks.forEach((task) => {
        const taskDate = new Date(plantingDate)
        taskDate.setDate(plantingDate.getDate() + task.daysFromPlanting)
        taskDate.setHours(0, 0, 0, 0) // Normalize task date

        // Check if the task date is today
        if (taskDate.getTime() === today.getTime()) {
          toast({
            title: `Reminder: ${task.task} Due Today!`,
            description: task.description,
            variant: "default",
          })
        }
      })
    }
  }, [actualPlantingDate, recommendation?.tasks, toast])

  return (
    <div className="min-h-screen gradient-bg">
      {/* Header */}
      <header className="bg-white/95 backdrop-blur-sm shadow-lg border-b border-primary-100 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-20">
            <div className="flex items-center space-x-3">
              <div className="bg-gradient-to-br from-primary-500 to-primary-600 p-3 rounded-2xl shadow-lg">
                <Leaf className="w-8 h-8 text-black" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-primary-600 to-primary-800 bg-clip-text text-transparent">
                  Smart Tomato Farming
                </h1>
                <p className="text-sm text-gray-600">AI-Powered Agriculture</p>
              </div>
            </div>
            <nav className="flex space-x-2">
              <Button
                variant={currentView === "welcome" ? "default" : "ghost"}
                onClick={resetToWelcome}
                className="font-semibold px-6 py-2 rounded-xl"
              >
                HOME
              </Button>
              <Button variant="ghost" className="font-semibold px-6 py-2 rounded-xl">
                CONTACT US
              </Button>
              <Button variant="ghost" className="font-semibold px-6 py-2 rounded-xl">
                LOGIN
              </Button>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {currentView === "welcome" && (
          <div className="space-y-16 animate-fade-in">
            {/* Hero Section */}
            <div className="text-center space-y-6">
              <div className="inline-flex items-center space-x-2 bg-primary-100 text-primary-800 px-4 py-2 rounded-full text-sm font-medium">
                <TrendingUp className="h-4 w-4" />
                <span>Advanced Agricultural Technology</span>
              </div>
              <h2 className="text-5xl md:text-6xl font-bold text-gray-900 leading-tight">
                Welcome to SmartTomato Farming{" "}
                <span className="bg-gradient-to-r from-primary-600 to-blue-600 bg-clip-text text-transparent">
                  Smart Tomato Farming
                </span>
              </h2>
              <p className="text-xl text-gray-600 max-w-4xl mx-auto leading-relaxed">
                Revolutionize your tomato farming with AI-powered insights. Get optimal planting schedules based on
                real-time weather data and detect pests instantly with our advanced machine learning technology.
              </p>
            </div>

            {/* Service Options */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-6xl mx-auto">
              {/* Optimal Schedule Option */}
              <Card className="card-hover border-0 shadow-2xl bg-gradient-to-br from-primary-50 to-primary-100 overflow-hidden">
                <CardHeader className="text-center pb-8 pt-12">
                  <div className="mx-auto bg-gradient-to-br from-primary-400 to-primary-600 p-6 rounded-3xl w-24 h-24 flex items-center justify-center mb-6 shadow-xl">
                    <Sprout className="h-12 w-12 text-white" />
                  </div>
                  <CardTitle className="text-3xl font-bold text-gray-900 mb-4">Optimal Planting Schedule</CardTitle>
                  <CardDescription className="text-lg text-gray-600 leading-relaxed">
                    Get weather-based recommendations for the perfect time to plant your tomatoes with our intelligent
                    forecasting system
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6 px-8 pb-12">
                  <ul className="space-y-4">
                    {[
                      { icon: Sun, text: "Real-time weather analysis" },
                      { icon: Calendar, text: "5-day forecast integration" },
                      { icon: Bell, text: "Automated farming schedule" },
                      { icon: Shield, text: "Task reminders and alerts" },
                    ].map((item, index) => (
                      <li key={index} className="flex items-center space-x-4">
                        <div className="bg-primary-100 p-2 rounded-lg">
                          <item.icon className="h-5 w-5 text-primary-600" />
                        </div>
                        <span className="text-gray-700 font-medium">{item.text}</span>
                      </li>
                    ))}
                  </ul>
                  <Button className="w-full btn-primary text-lg py-4 mt-8" onClick={() => setCurrentView("schedule")}>
                    Get Planting Schedule
                    <ArrowRight className="h-5 w-5 ml-2" />
                  </Button>
                </CardContent>
              </Card>

              {/* Pest Detection Option */}
              <Card className="card-hover border-0 shadow-2xl bg-gradient-to-br from-blue-50 to-indigo-50 overflow-hidden">
                <CardHeader className="text-center pb-8 pt-12">
                  <div className="mx-auto bg-gradient-to-br from-blue-400 to-blue-600 p-6 rounded-3xl w-24 h-24 flex items-center justify-center mb-6 shadow-xl">
                    <Bug className="h-12 w-12 text-white" />
                  </div>
                  <CardTitle className="text-3xl font-bold text-gray-900 mb-4">Pest Detection & Treatment</CardTitle>
                  <CardDescription className="text-lg text-gray-600 leading-relaxed">
                    Upload or capture images to instantly identify pests and receive comprehensive treatment
                    recommendations
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6 px-8 pb-12">
                  <ul className="space-y-4">
                    {[
                      { icon: Camera, text: "AI-powered pest identification" },
                      { icon: Upload, text: "Camera capture support" },
                      { icon: Shield, text: "Treatment recommendations" },
                      { icon: CheckCircle, text: "Prevention strategies" },
                    ].map((item, index) => (
                      <li key={index} className="flex items-center space-x-4">
                        <div className="bg-blue-100 p-2 rounded-lg">
                          <item.icon className="h-5 w-5 text-blue-600" />
                        </div>
                        <span className="text-gray-700 font-medium">{item.text}</span>
                      </li>
                    ))}
                  </ul>
                  <Button
                    className="w-full bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-semibold py-4 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 text-lg"
                    onClick={() => setCurrentView("detection")}
                  >
                    Detect Pests
                    <ArrowRight className="h-5 w-5 ml-2" />
                  </Button>
                </CardContent>
              </Card>
            </div>

            {/* Features Section */}
            <div className="bg-white/80 backdrop-blur-sm rounded-3xl p-12 shadow-2xl border border-primary-100">
              <h3 className="text-4xl font-bold text-center mb-4 text-gray-900">Why Choose Smart Tomato Farming?</h3>
              <p className="text-center text-gray-600 mb-12 text-lg max-w-3xl mx-auto">
                Our cutting-edge technology combines weather intelligence with AI-powered pest detection to maximize
                your harvest
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                {[
                  {
                    icon: Sun,
                    title: "Weather Integration",
                    description:
                      "Real-time weather data from OpenWeather API for accurate predictions and optimal timing",
                    color: "yellow",
                  },
                  {
                    icon: Camera,
                    title: "AI-Powered Detection",
                    description: "Advanced machine learning models for instant and accurate pest identification",
                    color: "purple",
                  },
                  {
                    icon: Bell,
                    title: "Smart Scheduling",
                    description: "Automated reminders and optimal timing for all your farming activities",
                    color: "green",
                  },
                ].map((feature, index) => (
                  <div key={index} className="text-center group">
                    <div
                      className={`bg-${feature.color}-100 p-4 rounded-2xl w-20 h-20 flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300`}
                    >
                      <feature.icon className={`h-10 w-10 text-${feature.color}-600`} />
                    </div>
                    <h4 className="font-bold text-xl mb-3 text-gray-900">{feature.title}</h4>
                    <p className="text-gray-600 leading-relaxed">{feature.description}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {currentView === "schedule" && (
          <div className="space-y-8 animate-fade-in">
            {/* Back Button */}
            <Button variant="outline" onClick={resetToWelcome} className="mb-6 btn-secondary bg-transparent">
              ← Back to Home
            </Button>

            {/* Location Input */}
            <Card className="shadow-xl border-0 bg-white/90 backdrop-blur-sm">
              <CardHeader className="pb-6">
                <CardTitle className="flex items-center space-x-3 text-2xl">
                  <div className="bg-primary-100 p-2 rounded-lg">
                    <MapPin className="h-6 w-6 text-primary-600" />
                  </div>
                  <span>Enter Your Location</span>
                </CardTitle>
                <CardDescription className="text-lg">
                  Select your location in Nigeria to get weather-based planting recommendations
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex space-x-4">
                  <div className="flex-1">
                    <Label htmlFor="location" className="text-base font-semibold">
                      State/Location
                    </Label>
                    <Input
                      id="location"
                      list="states"
                      value={location}
                      onChange={(e) => setLocation(e.target.value)}
                      placeholder="Enter your state or city"
                      className="mt-2 h-12 text-lg rounded-xl border-2 border-gray-200 focus:border-primary-500"
                    />
                    <datalist id="states">
                      {nigerianStates.map((state) => (
                        <option key={state} value={state} />
                      ))}
                    </datalist>
                  </div>
                  <Button
                    onClick={() => fetchWeatherData(location)}
                    disabled={!location || loading}
                    className="mt-8 btn-primary h-12 px-8"
                  >
                    {loading ? "Loading..." : "Get Weather"}
                  </Button>
                </div>
                {error && (
                  <Alert className="mt-6 border-red-200 bg-red-50">
                    <AlertTriangle className="h-5 w-5 text-red-600" />
                    <AlertDescription className="text-red-700 font-medium">{error}</AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>

            {/* Weather Display */}
            {weatherData && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <Card className="shadow-xl border-0 bg-white/90 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-3 text-xl">
                      <div className="bg-blue-100 p-2 rounded-lg">
                        <Thermometer className="h-6 w-6 text-blue-600" />
                      </div>
                      <span>Current Weather - {weatherData.location}</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-6">
                      <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-xl">
                        {getWeatherIcon(weatherData.current.condition)}
                        <span className="font-medium">{weatherData.current.condition}</span>
                      </div>
                      <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-xl">
                        <Thermometer className="h-5 w-5 text-red-500" />
                        <span className="font-medium">{weatherData.current.temp}°C</span>
                      </div>
                      <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-xl">
                        <Droplets className="h-5 w-5 text-blue-500" />
                        <span className="font-medium">{weatherData.current.humidity}% Humidity</span>
                      </div>
                      <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-xl">
                        <Wind className="h-5 w-5 text-gray-500" />
                        <span className="font-medium">{weatherData.current.windSpeed} km/h</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="shadow-xl border-0 bg-white/90 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-3 text-xl">
                      <div className="bg-purple-100 p-2 rounded-lg">
                        <Calendar className="h-6 w-6 text-purple-600" />
                      </div>
                      <span>5-Day Forecast</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {weatherData.forecast.map((day, index) => (
                        <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-xl">
                          <div className="flex items-center space-x-3">
                            {getWeatherIcon(day.condition)}
                            <span className="font-medium">{new Date(day.date).toLocaleDateString()}</span>
                          </div>
                          <div className="font-semibold text-gray-700">
                            {day.temp.min}°C - {day.temp.max}°C
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {recommendation && (
              <Card className="shadow-xl border-0 bg-white/90 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-3 text-2xl">
                    <CheckCircle
                      className={`h-7 w-7 ${recommendation.shouldPlant ? "text-primary-500" : "text-red-500"}`}
                    />
                    <span>Planting Recommendation</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    <Alert
                      className={`border-2 ${recommendation.shouldPlant ? "border-primary-200 bg-primary-50" : "border-red-200 bg-red-50"}`}
                    >
                      <AlertDescription>
                        <Badge
                          variant={recommendation.shouldPlant ? "default" : "destructive"}
                          className="text-sm px-4 py-1 font-semibold"
                        >
                          {recommendation.shouldPlant ? "RECOMMENDED" : "NOT RECOMMENDED"}
                        </Badge>
                        <p className="mt-3 text-base font-medium">{recommendation.reason}</p>
                      </AlertDescription>
                    </Alert>

                    {recommendation.shouldPlant && (
                      <div className="bg-primary-50 p-6 rounded-xl">
                        <h4 className="font-bold text-lg mb-4 text-primary-800">Optimal Planting Days:</h4>
                        <div className="flex flex-wrap gap-3">
                          {recommendation.optimalDays.map((day) => (
                            <Badge key={day} variant="outline" className="px-4 py-2 text-sm font-semibold bg-white">
                              {day}
                            </Badge>
                          ))}
                        </div>
                        {recommendation.optimalDays.length > 0 && !actualPlantingDate && (
                          <Button
                            onClick={() => handleConfirmPlanting(recommendation.optimalDays[0])} // Confirm the first optimal day for simplicity
                            className="mt-6 btn-primary"
                          >
                            Confirm Planting Date
                          </Button>
                        )}
                        {actualPlantingDate && (
                          <p className="mt-4 text-primary-700 font-medium">
                            Planting confirmed for: {actualPlantingDate}
                          </p>
                        )}
                      </div>
                    )}

                    <div>
                      <h4 className="font-bold text-lg mb-4 flex items-center space-x-3">
                        <div className="bg-blue-100 p-2 rounded-lg">
                          <Bell className="h-5 w-5 text-blue-600" />
                        </div>
                        <span>Farming Schedule & Reminders:</span>
                      </h4>
                      <div className="space-y-3">
                        {recommendation.tasks.map((task, index) => (
                          <div key={index} className="flex items-center space-x-4 p-4 bg-gray-50 rounded-xl">
                            <Badge variant="secondary" className="px-3 py-1 font-semibold">
                              Day {task.daysFromPlanting}
                            </Badge>
                            <div>
                              <span className="font-semibold text-gray-900">{task.task}</span>
                              <p className="text-gray-600 mt-1">{task.description}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        )}

        {currentView === "detection" && (
          <div className="space-y-8 animate-fade-in">
            {/* Back Button */}
            <Button variant="outline" onClick={resetToWelcome} className="mb-6 btn-secondary bg-transparent">
              ← Back to Home
            </Button>

            {!predictionResult ? (
              <Card className="shadow-xl border-0 bg-white/90 backdrop-blur-sm">
                <CardHeader className="pb-6">
                  <CardTitle className="text-2xl">Pest Detection & Identification</CardTitle>
                  <CardDescription className="text-lg">
                    Upload an image or take a photo of the pest to identify it and get treatment recommendations
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    {!imagePreview ? (
                      <div className="space-y-6">
                        <div className="flex flex-col sm:flex-row gap-4">
                          <Button
                            onClick={handleCameraCapture}
                            className="flex items-center justify-center space-x-3 btn-primary h-14 flex-1"
                            disabled={showCamera}
                          >
                            <Camera className="h-5 w-5" />
                            <span>{showCamera ? "Opening Camera..." : "Take Photo"}</span>
                          </Button>
                          <Label htmlFor="image-upload" className="cursor-pointer flex-1">
                            <Button asChild className="w-full h-14 btn-secondary">
                              <span className="flex items-center justify-center space-x-3">
                                <Upload className="h-5 w-5" />
                                <span>Upload Image</span>
                              </span>
                            </Button>
                          </Label>
                          <Input
                            id="image-upload"
                            type="file"
                            accept="image/jpeg,image/jpg,image/png"
                            onChange={handleImageUpload}
                            className="hidden"
                          />
                        </div>

                        {showCamera && (
                          <div className="bg-gradient-to-br from-primary-100 to-primary-200 p-12 rounded-2xl text-center">
                            <Camera className="h-16 w-16 mx-auto mb-6 text-primary-600" />
                            <p className="text-lg font-medium text-primary-800">Simulating camera access...</p>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="space-y-6">
                        <div className="relative bg-gray-100 rounded-2xl p-4">
                          <img
                            src={imagePreview || "/placeholder.svg"}
                            alt="Selected image"
                            className="max-w-full h-80 object-contain mx-auto rounded-xl"
                          />
                          <Button
                            onClick={resetPrediction}
                            size="sm"
                            variant="destructive"
                            className="absolute top-6 right-6 rounded-full p-2"
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </div>

                        <Button
                          onClick={handlePestPrediction}
                          disabled={predictionLoading}
                          className="w-full btn-primary h-14 text-lg"
                        >
                          {predictionLoading ? "Analyzing Image..." : "Identify Pest"}
                        </Button>
                      </div>
                    )}

                    {error && (
                      <Alert className="border-red-200 bg-red-50">
                        <AlertTriangle className="h-5 w-5 text-red-600" />
                        <AlertDescription className="text-red-700 font-medium">{error}</AlertDescription>
                      </Alert>
                    )}
                  </div>
                </CardContent>
              </Card>
            ) : (
              <Card className="shadow-xl border-0 bg-white/90 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center justify-between text-2xl">
                    <span>Pest Identification Result</span>
                    <Button
                      onClick={resetPrediction}
                      variant="outline"
                      size="sm"
                      className="btn-secondary bg-transparent"
                    >
                      <X className="h-4 w-4 mr-2" />
                      Cancel
                    </Button>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-8">
                    <div className="flex items-center space-x-6 p-6 bg-gray-50 rounded-2xl">
                      <img
                        src={imagePreview || "/placeholder.svg"}
                        alt="Analyzed image"
                        className="w-32 h-32 object-cover rounded-xl shadow-lg"
                      />
                      <div>
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">{predictionResult.pestName}</h3>
                        <Badge variant="default" className="text-base px-4 py-2 font-semibold">
                          Confidence: {predictionResult.confidence}%
                        </Badge>
                      </div>
                    </div>

                    <div className="bg-blue-50 p-6 rounded-2xl">
                      <h4 className="font-bold text-lg mb-3 text-blue-800">Description:</h4>
                      <p className="text-gray-700 leading-relaxed">{predictionResult.description}</p>
                    </div>

                    <div className="bg-primary-50 p-6 rounded-2xl">
                      <h4 className="font-bold text-lg mb-4 text-primary-800">Prevention Methods:</h4>
                      <ul className="space-y-2">
                        {predictionResult.prevention.map((method, index) => (
                          <li key={index} className="flex items-start space-x-3">
                            <CheckCircle className="h-5 w-5 text-primary-600 mt-0.5 flex-shrink-0" />
                            <span className="text-gray-700">{method}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div className="bg-orange-50 p-6 rounded-2xl">
                      <h4 className="font-bold text-lg mb-4 text-orange-800">Recommended Pesticides:</h4>
                      <ul className="space-y-2">
                        {predictionResult.pesticides.map((pesticide, index) => (
                          <li key={index} className="flex items-start space-x-3">
                            <Shield className="h-5 w-5 text-orange-600 mt-0.5 flex-shrink-0" />
                            <span className="text-gray-700">{pesticide}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div>
                      <h4 className="font-bold text-lg mb-4">Reference Images:</h4>
                      <div className="grid grid-cols-3 gap-4">
                        {predictionResult.images.map((image, index) => (
                          <img
                            key={index}
                            src={image || "/placeholder.svg"}
                            alt={`${predictionResult.pestName} reference ${index + 1}`}
                            className="w-full h-32 object-cover rounded-xl shadow-md"
                          />
                        ))}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        )}
      </main>
    </div>
  )
}




// export default function Home() {
//   return (
//     <div className="bg-red-500 text-white p-8 text-4xl font-bold">
//       RED BACKGROUND = TAILWIND WORKS!
//     </div>
//   )
// }
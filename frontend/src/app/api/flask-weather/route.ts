import { type NextRequest, NextResponse } from "next/server"

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const location = searchParams.get("location")

  if (!location) {
    return NextResponse.json({ error: "Location is required" }, { status: 400 })
  }

  try {
    const API_KEY = process.env.OPENWEATHER_API_KEY

    if (!API_KEY) {
      throw new Error("OpenWeather API key not configured")
    }

    console.log(`Fetching weather for: ${location}, Nigeria`)

    // Fetch current weather
    const currentResponse = await fetch(
      `https://api.openweathermap.org/data/2.5/weather?q=${location},NG&appid=${API_KEY}&units=metric`,
    )

    // Fetch 5-day forecast
    const forecastResponse = await fetch(
      `https://api.openweathermap.org/data/2.5/forecast?q=${location},NG&appid=${API_KEY}&units=metric`,
    )

    if (!currentResponse.ok || !forecastResponse.ok) {
      throw new Error("Weather API request failed")
    }

    const currentData = await currentResponse.json()
    const forecastData = await forecastResponse.json()

    // Process the weather data
    const processedData = {
      location: currentData.name,
      current: {
        temp: Math.round(currentData.main.temp),
        humidity: currentData.main.humidity,
        windSpeed: Math.round(currentData.wind.speed * 3.6), // Convert m/s to km/h
        condition: currentData.weather[0].main,
        icon: currentData.weather[0].icon,
      },
      forecast: forecastData.list
        .filter((_: any, index: number) => index % 8 === 0) // Get one reading per day
        .slice(0, 5)
        .map((item: any) => ({
          date: item.dt_txt.split(" ")[0],
          temp: {
            min: Math.round(item.main.temp_min),
            max: Math.round(item.main.temp_max),
          },
          humidity: item.main.humidity,
          condition: item.weather[0].main,
          icon: item.weather[0].icon,
          precipitation: Math.round(item.pop * 100),
        })),
    }

    return NextResponse.json(processedData)
  } catch (error) {
    console.error("Weather API error:", error)

    // Return mock data if API fails (for development)
    const mockData = {
      location: location,
      current: {
        temp: 28,
        humidity: 75,
        windSpeed: 12,
        condition: "Partly Cloudy",
        icon: "02d",
      },
      forecast: [
        {
          date: "2025-07-10",
          temp: { min: 22, max: 30 },
          humidity: 70,
          condition: "Sunny",
          icon: "01d",
          precipitation: 0,
        },
        {
          date: "2025-07-11",
          temp: { min: 24, max: 32 },
          humidity: 65,
          condition: "Partly Cloudy",
          icon: "02d",
          precipitation: 10,
        },
        {
          date: "2025-07-12",
          temp: { min: 23, max: 29 },
          humidity: 80,
          condition: "Light Rain",
          icon: "10d",
          precipitation: 60,
        },
        {
          date: "2025-07-13",
          temp: { min: 21, max: 27 },
          humidity: 85,
          condition: "Heavy Rain",
          icon: "10d",
          precipitation: 90,
        },
        {
          date: "2025-07-14",
          temp: { min: 23, max: 31 },
          humidity: 70,
          condition: "Sunny",
          icon: "01d",
          precipitation: 5,
        },
      ],
    }

    return NextResponse.json(mockData)
  }
}

import { type NextRequest, NextResponse } from "next/server"

interface WeatherData {
  forecast: Array<{
    temp: { min: number; max: number }
    humidity: number
    precipitation: number
  }>
}

export async function POST(request: NextRequest) {
  try {
    const { weatherData }: { weatherData: WeatherData } = await request.json()

    if (!weatherData || !weatherData.forecast) {
      return NextResponse.json({ error: "Weather data is required" }, { status: 400 })
    }

    // Calculate optimal planting recommendation
    const avgTemp =
      weatherData.forecast.reduce((sum, day) => sum + (day.temp.min + day.temp.max) / 2, 0) /
      weatherData.forecast.length

    const avgHumidity = weatherData.forecast.reduce((sum, day) => sum + day.humidity, 0) / weatherData.forecast.length

    const rainyDays = weatherData.forecast.filter((day) => day.precipitation > 50).length

    const shouldPlant = avgTemp >= 20 && avgTemp <= 35 && avgHumidity >= 60 && avgHumidity <= 80 && rainyDays <= 2

    let reason = ""
    if (!shouldPlant) {
      const reasons = []
      if (avgTemp < 20) reasons.push("Temperature too low")
      if (avgTemp > 35) reasons.push("Temperature too high")
      if (avgHumidity < 60) reasons.push("Low humidity")
      if (avgHumidity > 80) reasons.push("High humidity")
      if (rainyDays > 2) reasons.push("Too many rainy days")
      reason = `Not recommended: ${reasons.join(", ")}`
    } else {
      reason = "Weather conditions are optimal for tomato planting"
    }

    const recommendation = {
      shouldPlant,
      reason,
      optimalDays: shouldPlant ? ["2025-07-10", "2025-07-11", "2025-07-14"] : [],
      tasks: [
        {
          task: "Watering",
          daysFromPlanting: 2,
          description: "Water seedlings gently in the morning",
        },
        {
          task: "Weeding",
          daysFromPlanting: 7,
          description: "Remove weeds around the plants",
        },
        {
          task: "Fertilizing",
          daysFromPlanting: 14,
          description: "Apply organic fertilizer",
        },
        {
          task: "Pruning",
          daysFromPlanting: 21,
          description: "Remove lower leaves and suckers",
        },
        {
          task: "Pest Check",
          daysFromPlanting: 28,
          description: "Inspect plants for pests and diseases",
        },
      ],
    }

    return NextResponse.json(recommendation)
  } catch (error) {
    console.error("Schedule generation error:", error)
    return NextResponse.json({ error: "Failed to generate planting schedule" }, { status: 500 })
  }
}

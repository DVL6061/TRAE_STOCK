# Stock Prediction Frontend

This is the frontend application for the Stock Prediction System, an enterprise-grade stock forecasting system for the Indian stock market. It provides a user-friendly interface to interact with the prediction engine, visualize stock data, and analyze predictions.

## Features

- Real-time stock price monitoring and visualization
- Technical indicator analysis (RSI, MACD, EMA, SMA, Bollinger Bands, etc.)
- AI-powered stock price predictions with multiple timeframes
- Sentiment analysis of financial news
- Multilingual support (English and Hindi)
- Dark/Light theme support
- Responsive design for all devices

## Tech Stack

- React.js - Frontend framework
- React Router - Navigation and routing
- Chart.js - Data visualization
- i18next - Internationalization
- Tailwind CSS - Styling and UI components
- Axios - API communication
- React Icons - UI icons
- React Toastify - Notifications

## Project Structure

```
src/
├── components/       # Reusable UI components
│   ├── charts/       # Chart components
│   ├── common/       # Common UI elements
│   └── layout/       # Layout components (Header, Sidebar, Footer)
├── contexts/         # React contexts for state management
├── pages/            # Page components
├── App.js            # Main application component
├── i18n.js           # Internationalization configuration
├── index.css         # Global styles
├── index.js          # Application entry point
└── reportWebVitals.js # Performance monitoring
```

## Getting Started

### Prerequisites

- Node.js (v14 or later)
- npm or yarn

### Installation

1. Clone the repository
2. Navigate to the frontend directory
3. Install dependencies:

```bash
npm install
# or
yarn install
```

### Running the Development Server

```bash
npm start
# or
yarn start
```

The application will be available at http://localhost:3000

### Building for Production

```bash
npm run build
# or
yarn build
```

## Connecting to the Backend

The frontend is configured to connect to the backend API running at http://localhost:8000. This can be modified in the `package.json` file by changing the `proxy` field.

## Internationalization

The application supports English and Hindi languages. Language files are located in the `src/i18n.js` file.

## Theme Support

The application supports light and dark themes, which can be toggled in the header or in the settings page.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Disclaimer

This application is for educational and informational purposes only. It is not intended to provide investment advice. Always consult with a qualified financial advisor before making investment decisions.
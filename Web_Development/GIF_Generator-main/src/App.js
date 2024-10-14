import './index.css';
import Header from './components/Header' 
import RandomGIF from './components/RandomGIF' 
import GIFByTag from './components/GIFByTag' 
// import { apiurl } from process.env.REACT_APP_GIPHY_API_KEY

function App() {
  return (
    <div className="flex flex-col items-center gap-y-8 background w-[100vw] h-[100vh] overflow-x-hidden">
      <Header/>
      <RandomGIF/>
      <GIFByTag/>
    </div>
  );
}

export default App;

import React, { useContext } from 'react';
import { BiCalendar, BiMap, BiUser, BiBook, BiBriefcase, BiWorld, BiCalendarX } from 'react-icons/bi';
import { Context } from '../context/Context'; // Assuming your context file is located here

const About = () => {
  const { isDarkMode } = useContext(Context);

  return (
    <div className={`container p-4 ${isDarkMode ? 'text-white' : 'text-dark'}`} style={{ backgroundColor: isDarkMode ? '#333' : 'rgba(228, 193, 129, 0.272);' }}>
      <div className="d-flex flex-column flex-md-row mb-4 gap-4 align-items-center">
        <div className="col-md-6 text-center">
          <figure className="figure">
            <img src="chanakya.jpg" alt="Aacharya Chanakya Image" className="img-fluid rounded" style={{ maxWidth: '300px' }} />
            <figcaption className={`figure-caption mt-2 ${isDarkMode ? 'text-white' : 'text-dark'}`}>
              Chanakya (c. 350–283 BCE) was a philosopher, economist, and royal advisor.
            </figcaption>
          </figure>
        </div>
        <div className="col-md-6 d-flex flex-column gap-3">
          <h2 className="display-2 text-danger">About Aacharya Chanakya</h2>
          <p>
            Chanakya, also known as Kautilya or Vishnugupta, was an ancient Indian teacher, philosopher, economist, jurist, and royal advisor. He is traditionally recognized as the author of the ancient Indian political treatise, the Arthashastra.
          </p>
          <p>
            Born in 350 BCE in India, Chanakya played a pivotal role in the establishment of the Maurya Empire. He served as the chief advisor to both Emperor Chandragupta Maurya and his son, Bindusara.
          </p>
        </div>
      </div>

      <div className="container">
        <h6 className="display-6 text-warning my-2">Timeline of Aacharya Chanakya</h6>
        <div className="row">
          <div className="col-md-6">
            <ul className="list-group list-group-flush rounded shadow-md">
              <li className="list-group-item">
                <BiCalendar /> <strong>Born:</strong> 350 BCE
              </li>
              <li className="list-group-item">
                <BiMap /> <strong>Place of Birth:</strong> Takshashila, India
              </li>
              <li className="list-group-item">
                <BiUser /> <strong>Parents:</strong> Rishi Canak (Father)
              </li>
              <li className="list-group-item">
                <BiBook /> <strong>Education:</strong> Takshashila
              </li>
              <li className="list-group-item">
                <BiWorld /> <strong>Known For:</strong> Arthashastra, Nitishastra, Chanakya Neeti
              </li>
              <li className="list-group-item">
                <BiBriefcase /> <strong>Occupation:</strong> Teacher, Philosopher, Economist, Jurist, Advisor
              </li>
              <li className="list-group-item">
                <BiCalendarX /> <strong>Death:</strong> 283 BCE
              </li>
            </ul>
          </div>
          <div className="col-md-6 d-flex justify-content-center align-items-center text-center">
            <figure className="figure">
              <img
                className="figure-img img-fluid rounded"
                src="chanakya_and_chandragupta_maurya.jpg"
                alt="Chanakya with Chandragupta Maurya"
                style={{ maxWidth: '300px' }}
              />
              <figcaption className={`figure-caption m-2 text-center text-center ${isDarkMode ? 'text-white' : 'text-dark'}`}>
                Chanakya is often credited with playing a crucial role in the establishment of the Maurya Empire.
              </figcaption>
            </figure>
          </div>
        </div>
      </div>

      <div className="d-flex flex-column gap-4">
        <h6 className="display-6 text-warning">His Efforts</h6>
        <div className="row">
          <h4 className="text-success">Early Life & Education</h4>
          <figure className="figure mt-4">
            <img className="figure-img img-fluid rounded" src="takshashila.jpg" alt="Takshashila University" style={{width: '100%'}} />
            <figcaption className={`figure-caption mb-4 mt-3 text-center ${isDarkMode ? 'text-white' : 'text-dark'}`}>Chanakya received his education at the ancient Takshashila University, where he mastered various fields of knowledge.</figcaption>
          </figure>
          <p>Chanakya was born in 350 BCE in ancient India. He was a brilliant student and pursued his studies at the Takshashila University, one of the oldest universities in the world. He was proficient in various subjects, including economics, politics, and military strategy.</p>
        </div>

        <div className="row">
          <h4 className="text-success"><strong>Role in Establishing the Maurya Empire</strong></h4>
          <figure className="figure mt-4">
            <img className="figure-img rounded" src="Chandragupta_Maurya_Empire.png" alt="Maurya Empire" style={{ width: '100%' }} />
            <figcaption className={`figure-caption text-center text-center ${isDarkMode ? 'text-white' : 'text-dark'}`}>Chanakya's strategies were instrumental in the rise of the Maurya Empire, which became one of the largest empires in Indian history.</figcaption>
          </figure>
          <p>Chanakya played a crucial role in the rise of the Maurya Empire. He identified and mentored Chandragupta Maurya, helping him overthrow the Nanda dynasty. Chanakya's guidance was pivotal in Chandragupta's ascent to power, and he continued to serve as the chief advisor to both Chandragupta and his son, Bindusara.</p>
        </div>

        <div className="row">
          <h4 className="text-success"><strong>Arthashastra</strong></h4>
          <div className="col-md-6 text-center">
            <figure className="figure">
              <img src="arthashastra.webp" alt="Arthashastra" className="figure-img rounded" style={{ width: '300px' }} />
              <figcaption className={`figure-caption text-center ${isDarkMode ? 'text-white' : 'text-dark'}`}>The Arthashastra is Chanakya's most renowned work, covering topics such as economics, politics, military strategy, and statecraft.</figcaption>
            </figure>
          </div>
          <div className="col-md-6">
            <p>One of Chanakya's most significant contributions is the Arthashastra, an ancient Indian treatise on statecraft, economic policy, and military strategy. The Arthashastra provides comprehensive guidelines on governance, ethics, and administration, reflecting Chanakya's profound understanding of politics and economics.</p>
            <p>The Arthashastra consists of 15 books and covers a wide range of topics, including the duties of a king, the functions of government officials, the management of state finances, diplomacy, and warfare. It is considered one of the earliest works on economics and political science, offering insights that are still relevant today.</p>
          </div>
        </div>

        <div className="row">
          <h4 className="text-success"><strong>Chanakya Neeti</strong></h4>
          <div className="col-md-6 text-center">
            <img src="chanakya-statue.jpg" alt="Statue of Chanakya" className="img-fluid rounded" style={{ width: '365px' }} />
          </div>
          <div className="col-md-6">
            <h5 className="text-success">Leadership and Governance Principles</h5>
            <ul className="pl-4">
              <li><b>Focus on Strength:</b> Chanakya emphasized the necessity for rulers to prioritize building and maintaining military strength to defend and expand their domains.</li>
              <li><b>Importance of Allies:</b> He underscored the significance of forming robust alliances to bolster a kingdom's security and prosperity.</li>
              <li><b>Manage Resources:</b> Efficient resource management was central to Chanakya's governance philosophy, ensuring economic stability and growth.</li>
            </ul>
            <h5 className="text-success mt-4">Other Notable Aspects</h5>
            <ul className="pl-4">
              <li><b>Role in Chandragupta's Rise:</b> Chanakya's strategies were instrumental in Chandragupta Maurya's overthrow of the Nanda dynasty, leading to the establishment of the powerful Maurya Empire.</li>
              <li><b>Educational Contributions:</b> As a revered professor at Takshashila University, Chanakya contributed significantly to early Indian education and intellectual discourse.</li>
              <li><b>Diplomatic Skills:</b> Known for his diplomatic acumen, Chanakya forged strategic alliances crucial for expanding the Maurya Empire's influence across ancient India.</li>
              <li><b>Legacy and Influence:</b> Chanakya's seminal works, including the Arthashastra and Chanakya Neeti, continue to influence political thought and strategy worldwide.</li>
            </ul>
          </div>
        </div>
      </div>
      <div className="row">
        <h6 className="display-6 text-warning mt-4"><strong>Legacy</strong></h6>
        <p>Chanakya's legacy endures through his writings and the impact he had on Indian history. His teachings continue to influence modern economic and political thought. He is often regarded as the pioneer of classical economics and is considered one of the greatest political thinkers in history.</p>
      </div>

      <div id="carouselExample" className="carousel slide" data-bs-ride="carousel" data-bs-interval="4000">
        <div className="carousel-inner text-center p-4 mt-5" style={{ borderLeft: '5px solid rgb(220, 53, 69)', fontFamily: 'Georgia, serif', fontSize: '1.2em' }}>
          <div className="carousel-item active">
            "A person should not be too honest. Straight trees are cut first and honest people are screwed first."
          </div>
          <div className="carousel-item">
            "Learn from the mistakes of others. You can't live long enough to make them all yourselves."
          </div>
          <div className="carousel-item">
            "As soon as the fear approaches near, attack and destroy it."
          </div>
          <div className="carousel-item">
            "Education is the best friend. An educated person is respected everywhere. Education beats the beauty and the youth."
          </div>
          <div className="carousel-item">
            "Before you start some work, always ask yourself three questions – Why am I doing it, What the results might be, and Will I be successful. Only when you think deeply and find satisfactory answers to these questions, go ahead."
          </div>
          <div className="carousel-item">
            "Once you start working on something, don’t be afraid of failure and don’t abandon it. People who work sincerely are the happiest."
          </div>
          <div className="carousel-item">
            "The world’s biggest power is the youth and beauty of a woman."
          </div>
        </div>

        <button className="carousel-control-prev" type="button" data-bs-target="#carouselExample" data-bs-slide="prev">
          <span className="carousel-control-prev-icon" aria-hidden="true"></span>
          <span className="visually-hidden">Previous</span>
        </button>
        <button className="carousel-control-next" type="button" data-bs-target="#carouselExample" data-bs-slide="next">
          <span className="carousel-control-next-icon" aria-hidden="true"></span>
          <span className="visually-hidden">Next</span>
        </button>
      </div>
    </div>
  );
};

export default About;

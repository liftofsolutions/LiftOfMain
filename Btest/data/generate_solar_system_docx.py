"""
Script to create the 'The Solar System: An Overview' Word (.docx) document.
Run with:
    pip install python-docx
    python create_docx.py
"""

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

# Set standard page margins
for section in doc.sections:
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

# Title
title_p = doc.add_paragraph()
title_run = title_p.add_run("THE ARCHITECTURE OF OUR SOLAR SYSTEM:\nA JOURNEY ACROSS COSMIC NEIGHBORHOODS")
title_run.font.name = "Calibri"
title_run.font.size = Pt(20)
title_run.font.bold = True
title_run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
title_p.paragraph_format.space_before = Pt(0)
title_p.paragraph_format.space_after = Pt(14)
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

def add_heading(text, level=1):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(4)
    run = h.runs[0]
    run.font.name = "Calibri"
    if level == 1:
        run.font.size = Pt(14)
        run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    else:
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(0x33, 0x55, 0x88)
    return h

def add_body_paragraph(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(11)
    return p

def add_bullet(lead_text, body_text):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    lead = p.add_run(lead_text)
    lead.font.name = "Calibri"
    lead.font.size = Pt(11)
    lead.font.bold = True
    body = p.add_run(body_text)
    body.font.name = "Calibri"
    body.font.size = Pt(11)
    return p

# Introduction and Origins
add_heading("Introduction and Origins", level=1)
add_body_paragraph(
    "Our Solar System is a gravitationally bound cosmic community situated within the Orion–Cygnus Arm of the Milky Way "
    "galaxy, approximately 26,000 light-years from the galactic core. Formed roughly 4.6 billion years ago from the "
    "gravitational collapse of a giant interstellar molecular cloud, the Solar System represents an extraordinary tapestry "
    "of celestial dynamics, elemental differentiation, and physical diversity. The vast majority of the collapsing mass "
    "concentrated at the center to ignite the Sun, while the remaining flattened debris formed a protoplanetary disk from "
    "which planets, dwarf planets, moons, asteroids, and comets coalesced over millions of years."
)

# The Central Engine: The Sun
add_heading("The Central Engine: The Sun", level=1)
add_body_paragraph(
    "At the heart of this vast architecture resides the Sun, a yellow dwarf main-sequence star (spectral class G2V). "
    "Containing approximately 99.86 percent of the total mass of the entire Solar System, the Sun’s intense gravitational "
    "field dictates the orbital mechanics of every celestial body within several light-years."
)
add_body_paragraph(
    "In its core, immense temperatures exceeding 15 million degrees Celsius drive the thermonuclear fusion of hydrogen into "
    "helium, converting mass into radiant energy. This solar radiation bathes the system in light and heat, powering planetary "
    "climates, driving photochemical atmospheric reactions, and blowing the solar wind—a steady stream of charged particles "
    "that carves out a bubble known as the heliosphere."
)

# The Terrestrial Realm: The Inner Planets
add_heading("The Terrestrial Realm: The Inner Planets", level=1)
add_body_paragraph(
    "Orbiting closest to the Sun are the four terrestrial or rocky planets: Mercury, Venus, Earth, and Mars. These worlds "
    "are defined by solid surfaces, metallic cores primarily composed of iron and nickel, silicate mantles, and relatively "
    "high bulk densities."
)
add_bullet("Mercury: ", "The innermost planet is a heavily cratered, airless world experiencing extreme thermal volatility. Lacking a substantial atmosphere to trap heat, surface temperatures fluctuate wildly between daytime highs of roughly 430 degrees Celsius and nighttime lows plunging to minus 180 degrees Celsius. Despite its scorched exterior, radar observations have discovered water ice harbored within permanently shadowed polar craters.")
add_bullet("Venus: ", "Often referred to as Earth's twin in size and mass, Venus presents a radically inhospitable environment. Its dense atmosphere, composed mostly of carbon dioxide and cloaked in clouds of sulfuric acid, creates a runaway greenhouse effect. Surface temperatures hover around 465 degrees Celsius, making it hotter than Mercury. Venus also rotates in retrograde, turning exceptionally slowly on its axis.")
add_bullet("Earth: ", "The third rock from the Sun is currently the only known haven for life in the universe. Earth possesses a unique combination of moderate surface temperatures, liquid water oceans covering over 70 percent of its crust, an active tectonic cycle, and a nitrogen-oxygen atmosphere shielded by a dynamic geomagnetic field. Accompanied by its unusually large satellite, the Moon, Earth maintains a stabilized axial tilt that supports reliable seasonal cycles.")
add_bullet("Mars: ", "The Red Planet, tinted by oxidized iron minerals across its dusty surface, is a frigid desert world holding vital clues to planetary evolution. Mars hosts Olympus Mons, the largest volcano in the Solar System, and the massive canyon network of Valles Marineris. Ancient river valleys, alluvial fans, and subsurface ice deposits confirm that liquid water once flowed freely across its surface, making it the primary target for the search for past extraterrestrial microbial life.")

# The Cosmic Dividing Line: The Asteroid Belt
add_heading("The Cosmic Dividing Line: The Asteroid Belt", level=1)
add_body_paragraph(
    "Separating the terrestrial planets from the gas giants is the main Asteroid Belt, a sprawling ring of rocky and metallic "
    "remnants left over from the early solar nebula. Gravitational disruptions caused by the enormous mass of nearby Jupiter "
    "prevented these fragments from accreting into a fully formed planet. The belt contains millions of asteroids, ranging from "
    "microscopic dust specks to massive bodies like the dwarf planet Ceres, which harbors a salty mantle and potential "
    "subsurface water reserves."
)

# The Realm of the Giants: The Outer Planets
add_heading("The Realm of the Giants: The Outer Planets", level=1)
add_body_paragraph(
    "Beyond the asteroid belt lie the four giant planets, divided into two distinct classes: the gas giants and the ice giants."
)

add_heading("Gas Giants", level=2)
add_bullet("Jupiter: ", "The undisputed titan of the planetary family, Jupiter possesses more than twice the mass of all other planets combined. Composed predominantly of hydrogen and helium, it lacks a well-defined solid surface. Its turbulent atmosphere features alternating cloud bands and colossal storms, most notably the Great Red Spot, an anticyclonic storm that has raged for centuries. Jupiter governs an extensive satellite system, including the four Galilean moons: volcanic Io, icy ocean-bearing Europa, giant Ganymede, and cratered Callisto.")
add_bullet("Saturn: ", "Renowned for its breathtaking ring system made of billions of individual particles of water ice and rock, Saturn is the second-largest planet. It is remarkably light, possessing a density lower than that of liquid water. Saturn commands dozens of moons, including Titan—a planet-sized world with a dense nitrogen atmosphere and liquid methane-ethane lakes—and Enceladus, which sprays watery plumes from a subsurface ocean into space.")

add_heading("Ice Giants", level=2)
add_bullet("Uranus: ", "Classed alongside Neptune as an ice giant, Uranus contains higher proportions of \"ices\" such as water, ammonia, and methane surrounding a rocky core. Its pale cyan appearance results from methane gas absorbing red light. Uniquely, Uranus rotates nearly on its side with an axial tilt of 98 degrees, producing extreme, decades-long seasonal extremes.")
add_bullet("Neptune: ", "The outermost recognized major planet, Neptune is a deep-blue, dynamic world swept by supersonic winds that reach speeds up to 2,100 kilometers per hour. Like Uranus, its internal composition is rich in heavy volatile compounds. Its largest moon, Triton, orbits in retrograde and exhibits active cryovolcanism, erupting nitrogen geysers from beneath its frozen crust.")

# The Trans-Neptunian Frontier
add_heading("The Trans-Neptunian Frontier: Kuiper Belt and Dwarf Worlds", level=1)
add_body_paragraph(
    "Beyond the orbit of Neptune lies the Kuiper Belt, an icy realm populated by ancient remnants from the dawn of planet "
    "formation. Here reside several dwarf planets, including Pluto, which was reclassified from major planet status in 2006. "
    "Pluto features vast nitrogen ice plains, towering water-ice mountains, and an intricate relationship with its largest "
    "moon, Charon. Other Kuiper Belt denizens, such as Haumea, Makemake, and Eris, underline the dynamic diversity of this "
    "distant zone."
)

# The Outer Shell
add_heading("The Outer Shell: The Oort Cloud and Heliosphere", level=1)
add_body_paragraph(
    "The influence of the Sun extends far beyond the Kuiper Belt. The heliosphere pushes outward against interstellar space, "
    "forming a termination shock and heliopause where the solar wind yields to cosmic radiation. Farther out, extending potentially "
    "up to 100,000 astronomical units away, lies the hypothesized Oort Cloud: a spherical shell of billions of icy planetesimals. "
    "This distant cloud serves as the reservoir for long-period comets, periodically sending frozen relics plunging back into the "
    "inner Solar System when nudged by passing stars."
)

# Conclusion
add_heading("Conclusion", level=1)
add_body_paragraph(
    "The Solar System is not a static collection of isolated spheres, but a highly interconnected, constantly evolving "
    "celestial neighborhood. From the nuclear furnace of the Sun to the quiet, frozen periphery of the Oort Cloud, studying "
    "its diverse worlds reveals fundamental insights into how planetary systems form, evolve, and potentially foster life "
    "across the wider cosmos."
)

doc.save("solar_system.docx")
print("Saved solar_system.docx successfully!")